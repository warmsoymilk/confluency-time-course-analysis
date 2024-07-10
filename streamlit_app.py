import io
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

FOLDER_ID = '1PphA7iXH-_YMbLrZxB8wdtTLXcAyEuRh'
CREDENTIALS = 'ivynatal_tpu.json'

# Increments session state
def increment_step():
    st.session_state.update(step=st.session_state.step + 1)
    st.rerun()

# Function to convert timestamp into UTC datetime
# Timestamps are like: 2024-07-09T20:59:56.596Z
def timestamp_to_datetime(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)

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
    files.sort(key=lambda x: x['time'], reverse=True)

    # Filter for files ending with _well_confluencies.csv
    files = [file for file in files if file['name'].endswith('_well_confluencies.csv')]

    # Convert to pandas DataFrame
    files = pd.DataFrame(files)

    # Add a column to denote whether the file is selected
    files['selected'] = False

    # Save to Streamlit session state
    st.session_state.files = files

# Select files to include in analysis
def select_files_for_analysis():
    st.write('Select files to include in analysis:')
    files = st.session_state.files
    files = st.data_editor(
        files,
        column_config={
            'selected': 'Selected',
            'name': 'Filename',
            'time': 'Time Created',
        },
        disabled=['name', 'time'],
        column_order=['selected', 'name', 'time'],
        hide_index=True,
    )
    if len(files[files['selected']]) < 2:
        st.error('Please select at least two files to continue.')
    elif st.button('Continue'):
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
def download_selected_files():
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
        data[file['id']].drop('confluency_mean', axis=1, inplace=True)
        data[file['id']].rename(columns={'confluency_median': 'confluency'}, inplace=True)
        data[file['id']]['timepoint'] = file['timepoint_label']
    
    # Create a wells mapping dataframe
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

    # Save data to session state
    st.session_state.data = data
    st.session_state.wells = wells

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
        for cn in failed_cols:
            new_values = set(wells[cn].unique())
            old_values = set(wells_orig[cn].unique())
            extra_values = new_values - old_values
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

# Join data together
def join_data():
    data = st.session_state.data
    wells = st.session_state.wells
    for i, df in data.items():
        timepoint = df['timepoint'].iloc[0]
        df = pd.merge(df, wells[[timepoint, '__index']], left_on='well', right_on=timepoint, how='left')
        df = df.drop(columns=[timepoint])
        data[i] = df
    merged = pd.concat(data.values(), axis=0, ignore_index=True)
    merged = pd.merge(merged, wells[['__index', '__label']], on='__index', how='left')

    # Create a pivot table for plotting
    df = merged
    pivot_df = df.pivot(index='__label', columns='timepoint', values='confluency')
    pivot_df = pivot_df.fillna(0)  # Handle missing values

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

    # Display the plot
    st.pyplot(plt)

# Plot data
def plot_data():
    st.write('')

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0

# Write out title (this never changes)
st.title('Confluency Time Course Analysis')

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
    join_data()