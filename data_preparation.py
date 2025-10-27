import pandas as pd
import json
import numpy as np

# File path
BORG_TRACE_PATH = r"C:\Users\Shobit\PycharmProjects\Hackathons\Predictive Checkpointing and Proactive Fault Tolerance in Distributed Systems\Google_brog\borg_traces_data.csv"

# Output path
OUTPUT_PATH = r"C:\Users\Shobit\PycharmProjects\Hackathons\Predictive Checkpointing and Proactive Fault Tolerance in Distributed Systems\processed_borg_data.csv"

CHUNK_SIZE = 500_000

def parse_resource_request(rr):
    """
    Parse the 'resource_request' JSON-like string into a dictionary.
    Example format: '{"cpus": 2.0, "memory": 4096}'
    """
    try:
        if isinstance(rr, str):
            return json.loads(rr.replace("'", "\""))
        elif isinstance(rr, dict):
            return rr
        else:
            return {}
    except Exception:
        return {}

def process_chunk(chunk):
    """Cleans and extracts CPU, Memory, and failure info per chunk safely."""
    # Drop irrelevant columns to reduce memory
    columns_to_keep = [
        'time', 'machine_id', 'resource_request', 'average_usage',
        'maximum_usage', 'cpu_usage_distribution', 'event', 'failed'
    ]
    chunk = chunk.loc[:, [col for col in columns_to_keep if col in chunk.columns]].copy()

    # Parse JSON resource requests safely
    chunk.loc[:, 'resource_request'] = chunk['resource_request'].apply(parse_resource_request)
    chunk.loc[:, 'cpu'] = chunk['resource_request'].apply(lambda x: x.get('cpus', np.nan) if isinstance(x, dict) else np.nan)
    chunk.loc[:, 'memory'] = chunk['resource_request'].apply(lambda x: x.get('memory', np.nan) if isinstance(x, dict) else np.nan)

    # Convert numeric safely
    chunk.loc[:, 'cpu'] = pd.to_numeric(chunk['cpu'], errors='coerce').fillna(0)
    chunk.loc[:, 'memory'] = pd.to_numeric(chunk['memory'], errors='coerce').fillna(0)

    # Ensure 'time' is datetime
    chunk.loc[:, 'time'] = pd.to_datetime(chunk['time'], errors='coerce')

    # Keep only valid rows
    chunk = chunk.dropna(subset=['machine_id', 'time'])

    return chunk

def process_borg_trace(file_path):
    """Read Borg dataset in chunks and process efficiently."""
    processed_chunks = []
    print(f"Processing Borg trace file in chunks from: {file_path}")
    print(f"Chunk size: {CHUNK_SIZE:,} rows")

    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)):
        print(f"Processing chunk {i + 1}...")
        processed_chunk = process_chunk(chunk)
        processed_chunks.append(processed_chunk)

    # Combine chunks
    final_df = pd.concat(processed_chunks, ignore_index=True)

    # Sort by machine and time for time-series usage
    final_df = final_df.sort_values(['machine_id', 'time']).reset_index(drop=True)

    # Fill NaNs
    final_df = final_df.fillna(0)

    # Save processed data
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Processed data saved to {OUTPUT_PATH}")
    print(f"✅ Total rows processed: {len(final_df):,}")

    return final_df

if __name__ == "__main__":
    process_borg_trace(BORG_TRACE_PATH)
