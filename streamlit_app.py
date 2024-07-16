import io
from datetime import datetime
from dateutil import parser, tz

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

FOLDER_ID = '1PphA7iXH-_YMbLrZxB8wdtTLXcAyEuRh'

# Increments session state
def increment_step(step_to=None):
    if step_to is None:
        step_to = st.session_state.step + 1
    st.session_state.update(step=step_to)
    st.rerun()

# Function to convert timestamp into UTC datetime
# Timestamps are like: 2024-07-09T20:59:56.596Z
def timestamp_to_datetime(ts):
    datetime_obj = parser.parse(ts)
    return datetime_obj

# Converts datetime object to PST human-readable string
def datetime_to_human_readable(dt):
    pst_timezone = tz.gettz('America/Los_Angeles')
    datetime_obj_pst = dt.astimezone(pst_timezone)
    human_readable = datetime_obj_pst.strftime('%Y-%m-%d %I:%M:%S %p %Z')
    return human_readable

# Return authenticated Google Drive connectoion
def get_drive_service():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets['service_account'],
        scopes=['https://www.googleapis.com/auth/drive']
    )
    drive_service = build('drive', 'v3', credentials=credentials)
    return drive_service

# Download list of files from Google Drive
def download_file_list():
    # Get list of files
    drive_service = get_drive_service()
    query = f"'{FOLDER_ID}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name, createdTime)", supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
    files = results.get('files', [])

    # Add UTC timestamp to each entry and sort from most recent
    for file in files:
        file['time'] = timestamp_to_datetime(file['createdTime'])
        file['time_str'] = datetime_to_human_readable(file['time'])
    files.sort(key=lambda x: x['time'], reverse=True)

    # Filter for files ending with _well_confluencies.csv
    files = [file for file in files if file['name'].endswith('_well_confluencies.csv')]

    # Convert to pandas DataFrame
    files = pd.DataFrame(files)

    # Deduplicate files with the same filename, keeping the one with the most recent timestamp
    files = files.sort_values('time', ascending=False).drop_duplicates('name').sort_index()

    # Remove anything in name after _BF_LED_
    files['name'] = files['name'].str.split('_BF_LED_').str[0]

    # Add a column to denote whether the file is selected
    files['selected'] = False

    # Save to Streamlit session state
    st.session_state.files = files

# Select files to include in analysis
def select_files_for_analysis():
    st.write('Select either a single file to view the per-well data for that plate only, or select multiple files to perform a time-course analysis.')
    files = st.session_state.files
    files = st.data_editor(
        files,
        column_config={
            'selected': 'Selected',
            'name': 'Filename',
            'time_str': 'Time Created',
        },
        disabled=['name', 'time_str'],
        column_order=['selected', 'name', 'time_str'],
        hide_index=True,
    )
    num_selected = len(files[files['selected']])
    if num_selected == 0:
        st.error('Please select at least one file to continue.')
    elif num_selected == 1 and st.button('Continue'):
        files = files[files['selected']]
        st.session_state.files = files
        increment_step(10)
    elif num_selected >= 2 and st.button('Continue'):
        files = files[files['selected']]
        st.session_state.files = files
        increment_step()

# Allow user to edit default timepoint labels
def edit_timepoint_labels():
    # Limit to files that were selected
    files = st.session_state.files

    # Insert default timepoint_label column if it doesn't exist
    if 'timepoint_label' not in files.columns:
        files.insert(0, 'timepoint_label', [str(i) for i in range(1, len(files) + 1)[::-1]])
    
    # Ask user to edit labels
    st.write('Please review the autogenerated timepoint labels and make any necessary changes. (Double-click on a label to edit it.) User-entered values must be numeric.')
    files = st.data_editor(
        files,
        column_config={
            'name': 'Filename',
            'time': 'Time Created',
            'timepoint_label': 'Timepoint',
        },
        disabled=['name', 'time'],
        column_order=['name', 'time', 'timepoint_label'],
        hide_index=True,
    )
    if st.button('Continue'):
        st.session_state.files = files
        increment_step()

# Downloads the selected files from Google Drive
def download_selected_files(no_timepoint_label=False):
    # Get list of files
    files = st.session_state.files

    # Get Google Drive service
    drive_service = get_drive_service()

    # Download each file
    i = 0
    data = {}
    for _, file in files.iterrows():
        i += 1
        st.write(f'Downloading file {i} of {len(files)}: {file["name"]}...')
        request = drive_service.files().get_media(fileId=file['id'])
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        file_io.seek(0)
        data[file['id']] = pd.read_csv(file_io)
        if 'confluency_mean' in data[file['id']].columns:
            data[file['id']].drop('confluency_mean', axis=1, inplace=True)
        data[file['id']].rename(columns={'confluency_median': 'confluency'}, inplace=True)
        if not no_timepoint_label:
            data[file['id']]['timepoint'] = file['timepoint_label']
    
    # Create a wells mapping dataframe
    if not no_timepoint_label:
        wells = {}
        max_wells = max(df.shape[0] for df in data.values())
        for df in data.values():
            df_wells = df['well'].tolist()
            if len(df_wells) < max_wells:
                df_wells += [''] * (max_wells - len(df_wells))
            wells[df['timepoint'].iloc[0]] = df_wells
        wells = pd.DataFrame(wells)

        # Add label column to wells
        lowest_timestamp = str(min(int(cn) for cn in wells.columns))
        wells['__label'] = wells[lowest_timestamp].apply(lambda x: 'Well ' + x)
        st.session_state.wells = wells

    # Save data to session state
    st.session_state.data = data

# Allow user to edit well mapping
def edit_well_mapping():
    wells = st.session_state.wells
    wells_orig = wells.copy()
    st.write('Please review the autogenerated cross-timestamp well mapping and make any necessary changes. (Double-click on a label to edit it.)')
    col_names = {cn: 'Timepoint ' + cn for cn in wells.columns}
    col_names['__label'] = 'Label'
    wells = st.data_editor(
        wells,
        column_config=col_names,
        column_order=['__label'] + sorted(cn for cn in wells.columns if cn != '__label'),
        hide_index=True,
    )
    failed_cols = []
    for cn in wells.columns:
        if cn == '__label':
            continue
        new_values = sorted(wells[cn].unique())
        old_values = sorted(wells_orig[cn].unique())
        if len(new_values) != len(old_values) or any(new_values[i] != old_values[i] for i in range(len(new_values))):
            failed_cols.append(cn)
    if len(failed_cols) > 0:
        error_msg = 'Please ensure that all wells in the original data are present in the new mapping.\n\n'
        for cn in failed_cols[::-1]:
            new_values = set(wells[cn].unique())
            old_values = set(wells_orig[cn].unique())
            extra_values = new_values - old_values
            extra_values = set(x for x in extra_values if x != '')
            missing_values = old_values - new_values
            if len(extra_values) > 0:
                error_msg += 'Timepoint ' + cn + ' has erroneous values: ' + ', '.join(extra_values) + '  \n'
            if len(missing_values) > 0:
                error_msg += 'Timepoint ' + cn + ' is missing wells: ' + ', '.join(missing_values) + '  \n'
            error_msg += '\n\n'
        st.error(error_msg)
    else:
        if st.button('Continue'):
            wells['__index'] = range(wells.shape[0])
            st.session_state.wells = wells
            increment_step()

# Prepare for plotting data
def prepare_for_plotting_time_course():
    data = st.session_state.data
    wells = st.session_state.wells
    for i, df in data.items():
        timepoint = df['timepoint'].iloc[0]
        df = pd.merge(df, wells[[timepoint, '__index']], left_on='well', right_on=timepoint, how='left')
        df = df.drop(columns=[timepoint])
        data[i] = df
    df = pd.concat(data.values(), axis=0, ignore_index=True)
    df = pd.merge(df, wells[['__index', '__label']], on='__index', how='left')

    # Create a pivot table for plotting
    pivot_df = df.pivot(index='__label', columns='timepoint', values='confluency')
    pivot_df = pivot_df.fillna(0)  # Handle missing values

    # Save plot to session state
    st.session_state.pivot_df = pivot_df

# Actually show the plot
def show_time_course_plot():
    pivot_df = st.session_state.pivot_df

    # Define the bar width and positions
    num_timepoints = len(pivot_df.columns)
    bar_width = 0.8 / num_timepoints  # Adjust bar width based on number of timepoints
    labels = pivot_df.index
    x = np.arange(len(labels))

    # Create the bar plots
    fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size for better readability
    for idx, column in enumerate(pivot_df.columns):
        ax.bar(x + idx * bar_width, pivot_df[column], bar_width, label=column)

    # Add labels, title, and legend
    ax.set_ylabel('Confluency')
    ax.set_xticks(x + bar_width * (num_timepoints - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend(title='Timepoint', bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside the plot

    # Display the plot
    plt.tight_layout()
    st.write('Here is a bar chart showing the time course of confluency for each well:')
    st.pyplot(plt)

    if st.button('Restart analysis'):
        increment_step(0)

# Plot data for a single file
def plot_single_file():
    df = list(st.session_state.data.values())[0]

    wells = df['well'].tolist()
    well_rows = [ord(well[0]) for well in wells]
    well_cols = [int(well[1:]) for well in wells]
    rows = list(chr(i) for i in range(min(well_rows), max(well_rows) + 1))
    cols = list(str(i) for i in range(min(well_cols), max(well_cols) + 1))

    # Create a dictionary to map well names to their confluency_median values
    well_confluency = {row['well']: row['confluency'] for _, row in df.iterrows()}

    # Create a matrix to hold the confluency values
    confluency_matrix = np.full((len(rows), len(cols)), -1.0)

    # Populate the confluency matrix
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            well_name = f"{row}{col}"
            if well_name in well_confluency:
                confluency_matrix[i, j] = well_confluency[well_name]

    # Create a custom colormap from white to red
    cmap = LinearSegmentedColormap.from_list('white_red', ['white', 'red'], N=256)

    # Plot the grid
    fig, ax = plt.subplots(figsize=(len(cols), len(rows)))

    # Display the confluency matrix using the colormap
    cax = ax.imshow(confluency_matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    # Plot the grid
    fig, ax = plt.subplots(figsize=(len(cols), len(rows)))

    # Display the confluency matrix using the colormap
    cax = ax.imshow(confluency_matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    # Set the ticks and labels
    ax.set_xticks(np.arange(len(cols)) + 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows)) + 0.5, minor=True)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)

    # Set the grid lines
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

    # Hide the tick marks
    ax.tick_params(which='major', size=0)
    ax.tick_params(which='minor', size=0)

    # Optionally, add labels to each cell
    for i in range(len(rows)):
        for j in range(len(cols)):
            if confluency_matrix[i, j] != -1.0:
                ax.text(j, i, f"{confluency_matrix[i, j]:.2f}", ha='center', va='center', color='black' if confluency_matrix[i, j] < 0.5 else 'white')

    # Add a colorbar to show the gradient
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Confluency (median across tiles)')

    # Show the plot
    plt.tight_layout()
    st.write(f'Per-well confluencies for the selected plate ({st.session_state.files["name"].iloc[0]}):')
    st.pyplot(plt)

    if st.button('Restart analysis'):
        increment_step(0)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0

# Write out title (this never changes)
st.set_page_config(page_title='Confluency Data Analysis')
st.title('Confluency Data Analysis')

if st.session_state.step == 0:
    download_file_list()
    increment_step()
if st.session_state.step == 1:
    select_files_for_analysis()
elif st.session_state.step == 2:
    edit_timepoint_labels()
elif st.session_state.step == 3:
    download_selected_files()
    increment_step()
elif st.session_state.step == 4:
    edit_well_mapping()
elif st.session_state.step == 5:
    prepare_for_plotting_time_course()
    increment_step()
elif st.session_state.step == 6:
    show_time_course_plot()
elif st.session_state.step == 10:
    download_selected_files(no_timepoint_label=True)
    increment_step()
elif st.session_state.step == 11:
    plot_single_file()