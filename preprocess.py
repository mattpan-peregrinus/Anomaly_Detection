import os 
import zipfile
import pandas as pd
import torch
from torch_geometric.data import HeteroData

def preprocess_data(zip_path, save_path):
    extracted_folder = 'extracted_data'
    
    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)
        
    # Load parquet files 
    transactions_df = pd.read_parquet(os.path.join(extracted_folder, 'transactions.parquet'))
    token_transfers_df = pd.read_parquet(os.path.join(extracted_folder, 'token_transfers.parquet'))
    dex_swaps_df = pd.read_parquet(os.path.join(extracted_folder, 'dex_swaps.parquet'))
    
    data = HeteroData()
    
    # Add nodes 
    eoa_addresses = transactions_df['FROM_ADDRESS'].unique()
    eoa_mapping = {address: i for i, address in enumerate(eoa_addresses)}
    data['EOA Address'].x = torch.randn(len(eoa_addresses), 16)
    
    token_contracts = token_transfers_df['TO_ADDRESS'].unique()
    token_mapping = {address: i for i, address in enumerate(token_contracts)}
    data['Token Contract'].x = torch.randn(len(token_contracts), 16)
    
    dex_addresses = dex_swaps_df['ORIGIN_TO_ADDRESS'].unique()
    dex_mapping = {address: i for i, address in enumerate(dex_addresses)}
    data['DEX'].x = torch.rand(len(dex_addresses), 16)
    
    lp_addresses = dex_swaps_df['CONTRACT_ADDRESS'].unique()
    lp_mapping = {address: i for i, address in enumerate(lp_addresses)}
    data['Liquidity Provider'].x = torch.rand(len(lp_addresses), 16)
    
    transactions = transactions_df['TX_HASH'].unique()
    transactions_mapping = {address: i for i, address in enumerate(transactions)}
    data['Transaction'].x = torch.rand(len(transactions), 16)
    
    transfers = token_transfers_df[['TX_HASH', 'EVENT_INDEX']].drop_duplicates()
    transfers['Transfer_ID'] = transfers['TX_HASH'] + "_" + transfers['EVENT_INDEX'].astype(str)
    transfer_mapping = {row.Transfer_ID: i for i, row in transfers.iterrow()}
    data['Transfer'].x = torch.rand(len(transfers), 16)
    
    
    # Add edges    
    # Case 1: A transfers ETH to B
    data['EOA Address', 'sends', 'Transaction'].edge_index = torch.tensor([
        [eoa_mapping[from_addr] for from_addr in transactions_df['FROM_ADDRESS']],
        [transactions_mapping[tx] for tx in transactions_df['TX_HASH']]
    ], dtype=torch.long)
    
    data['Transaction', 'sent_to', 'EOA Address'].edge_index = torch.tensor([
        [transactions_mapping[tx] for tx in transactions_df['TX_HASH']],
        [eoa_mapping[to_addr] for to_addr in transactions_df['TO_ADDRESS']] # Needs to add this 
        ], dtype=torch.long)

    data['Transaction', 'contains', 'Transfer'].edge_index = torch.tensor([
        [transactions_mapping[tx] for tx in transactions_df['TX_HASH']],
        [transfer_mapping[tx] for tx in transactions_df['TX_HASH']]  # Needs to fix this
    ], dtype=torch.long)

    data['EOA Address', 'sends', 'Transfer'].edge_index = torch.tensor([
        [eoa_mapping[from_addr] for from_addr in transactions_df['FROM_ADDRESS']],
        [transfer_mapping[tx] for tx in transactions_df['TX_HASH']]  # Needs to fix this
    ])
    
    data['Transfer', 'sent_to', 'EOA Address'].edge_index = torch.tensor([
        [transfer_mapping[tx] for tx in transactions_df['TX_HASH']],
        [eoa_mapping[to_addr] for to_addr in transactions_df['TO_ADDRESS']] # Needs to add this
    ], dtype=torch.long)
    
    
    
    # Case 2: A transfers a token to B 
    data['EOA Address', 'sends', 'Transaction'].edge_index = torch.tensor([
        [eoa_mapping[from_addr] for from_addr in token_transfers_df['FROM_ADDRESS']],
        [transactions_mapping[tx] for tx in token_transfers_df['TX_HASH']]
    ], dtype=torch.long)

    data['Transaction', 'sent_to', 'Token Contract'].edge_index = torch.tensor([
        [transactions_mapping[tx] for tx in token_transfers_df['TX_HASH']],
        [token_mapping[to_addr] for to_addr in token_transfers_df['TO_ADDRESS']]
    ], dtype=torch.long)

    data['Transaction', 'contains', 'Transfer'].edge_index = torch.tensor([
        [transactions_mapping[tx] for tx in token_transfers_df['TX_HASH']],
        [transfer_mapping[row.TX_HASH + "_" + str(row.EVENT_INDEX)] for _, row in token_transfers_df.iterrows()]
    ], dtype=torch.long)

    data['Transfer', 'includes', 'Token Contract'].edge_index = torch.tensor([
        [transfer_mapping[row.TX_HASH + "_" + str(row.EVENT_INDEX)] for _, row in token_transfers_df.iterrows()],
        [token_mapping[to_addr] for to_addr in token_transfers_df['TO_ADDRESS']]
    ], dtype=torch.long)

    data['Transfer', 'sent_to', 'EOA Address'].edge_index = torch.tensor([
        [transfer_mapping[row.TX_HASH + "_" + str(row.EVENT_INDEX)] for _, row in token_transfers_df.iterrows()],
        [eoa_mapping[to_addr] for to_addr in token_transfers_df['TO_ADDRESS']]
    ], dtype=torch.long)
    
    
    
    # Case 3: A swaps token A for token B on DEX
    data['EOA Address', 'sends', 'Transaction'].edge_index = torch.tensor([
        [eoa_mapping[from_addr] for from_addr in dex_swaps_df['ORIGIN_TO_ADDRESS']],
        [transactions_mapping[tx] for tx in dex_swaps_df['TX_HASH']]
    ], dtype=torch.long)

    data['Transaction', 'sent_to', 'DEX'].edge_index = torch.tensor([
        [transactions_mapping[tx] for tx in dex_swaps_df['TX_HASH']],
        [dex_mapping[dex] for dex in dex_swaps_df['CONTRACT_ADDRESS']]
    ], dtype=torch.long)

    data['Transaction', 'contains', 'Transfer'].edge_index = torch.tensor([
        [transactions_mapping[tx] for tx in dex_swaps_df['TX_HASH']],
        [transfer_mapping[tx] for tx in dex_swaps_df['TOKEN_X']]
    ], dtype=torch.long)

    data['Transfer', 'includes', 'Token Contract'].edge_index = torch.tensor([
        [transfer_mapping[tx] for tx in dex_swaps_df['TOKEN_X']],
        [token_mapping[token_x] for token_x in dex_swaps_df['TOKEN_X']]
    ], dtype=torch.long)

    data['Transfer', 'sent_to', 'Liquidity Provider'].edge_index = torch.tensor([
        [transfer_mapping[tx] for tx in dex_swaps_df['TOKEN_X']],
        [lp_mapping[lp] for lp in dex_swaps_df['CONTRACT_ADDRESS']]
    ], dtype=torch.long)

    data['Liquidity Provider', 'sends', 'Transfer'].edge_index = torch.tensor([
        [lp_mapping[lp] for lp in dex_swaps_df['CONTRACT_ADDRESS']],
        [transfer_mapping[tx] for tx in dex_swaps_df['TOKEN_Y']]
    ], dtype=torch.long)

    data['Transfer', 'includes', 'Token Contract'].edge_index = torch.tensor([
        [transfer_mapping[tx] for tx in dex_swaps_df['TOKEN_Y']],
        [token_mapping[token_y] for token_y in dex_swaps_df['TOKEN_Y']]
    ], dtype=torch.long)

    data['Transfer', 'sent_to', 'EOA Address'].edge_index = torch.tensor([
        [transfer_mapping[tx] for tx in dex_swaps_df['TOKEN_Y']],
        [eoa_mapping[to_addr] for to_addr in dex_swaps_df['ORIGIN_TO_ADDRESS']]
    ], dtype=torch.long)
    
    
    
    # Save the graph
    torch.save(data, save_path)
    print('Graph data saved to', save_path)

if __name__ == '__main__':
    zip_path = 'data/spam_token_prediction.zip'
    save_path = 'data/graph_data.pt'
    preprocess_data(zip_path, save_path)
    