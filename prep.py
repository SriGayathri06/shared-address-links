from pathlib import Path
import pandas as pd
import numpy as np


# ---------- CONFIG ----------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INPUT = DATA_DIR / "Cleaned_Full_Address_Contributions.csv"   # source csv,  <- use your cleaned file path
OUT_DIR = DATA_DIR                                            # or BASE_DIR / "artifacts"
N_MIN_AT_ADDRESS = 2                                        # min distinct contributors at an address
# ---------------------------


df = pd.read_csv(INPUT)
# clean the Amount column as it is formatted as $
df["Amount"] = (
    df["Amount"]
    .astype(str)                 # make sure it’s string
    .str.replace("$", "", regex=False)  # remove dollar sign
    .str.replace(",", "", regex=False)  # remove commas
    .astype(float)               # convert to float
)


# sanity: required cols
required = ["Contributor Name", "Contributor Type", "full_address"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# defaults
if "Amount" not in df.columns:
    df["Amount"] = 0.0
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# exact grouping by address (use your existing 'group_id' if already present)
if "group_id" not in df.columns:
    df["group_id"] = pd.factorize(df["full_address"])[0]

# stats per address
addr_stats = (df.groupby(["group_id", "full_address"], as_index=False)
                .agg(
                    contributors=("Contributor Name", lambda s: pd.Series(s.unique()).size),
                    total_amount=("Amount", "sum"),
                    tx_count=("Contributor Name", "size")
                ))

# keep addresses with >= N_MIN_AT_ADDRESS distinct contributors
shared_addr_ids = set(addr_stats.loc[addr_stats["contributors"] >= N_MIN_AT_ADDRESS, "group_id"])
df_shared = df[df["group_id"].isin(shared_addr_ids)].copy()

# stable ids
df_shared["contrib_id"] = "person:" + pd.factorize(df_shared["Contributor Name"])[0].astype(str)
df_shared["address_id"] = "addr:" + df_shared["group_id"].astype(str)

# contributor nodes
nodes_contrib = (df_shared.groupby(["contrib_id", "Contributor Name", "Contributor Type"], as_index=False)
                 .agg(total_amount=("Amount", "sum"),
                      tx_count=("full_address", "size")))

nodes_contrib.rename(columns={"Contributor Name": "label",
                              "Contributor Type": "contrib_type"}, inplace=True)
nodes_contrib["type"] = "contributor"

# address nodes
nodes_addr = (df_shared.groupby(["address_id", "full_address"], as_index=False)
              .agg(contributors=("Contributor Name", lambda s: pd.Series(s.unique()).size),
                   total_amount=("Amount", "sum"),
                   tx_count=("full_address", "size")))
nodes_addr.rename(columns={"full_address": "label"}, inplace=True)
nodes_addr["type"] = "address"
nodes_addr["contrib_type"] = np.nan  # n/a

# combine
nodes = pd.concat([
    nodes_contrib.rename(columns={"contrib_id": "id"})[["id", "label", "type", "contrib_type", "total_amount", "tx_count"]],
    nodes_addr.rename(columns={"address_id": "id"})[["id", "label", "type", "contrib_type", "total_amount", "tx_count"]],
], ignore_index=True)

# edges: contributor -> address (bipartite)
edges = (df_shared.groupby(["contrib_id", "address_id", "full_address"], as_index=False)
         .agg(tx_count=("Contributor Name", "size"),
              total_amount=("Amount", "sum")))
edges.rename(columns={"contrib_id": "source", "address_id": "target"}, inplace=True)
edges["edge_type"] = "at_address"
edges["address"] = edges["full_address"]
edges = edges[["source", "target", "edge_type", "address", "tx_count", "total_amount"]]

# top shared addresses table for quick insights
top_shared = (addr_stats[addr_stats["group_id"].isin(shared_addr_ids)]
              .sort_values(["contributors", "total_amount"], ascending=[False, False]))

OUT_DIR.mkdir(parents=True, exist_ok=True)
(nodes).to_csv(OUT_DIR / "nodes.csv", index=False)
(edges).to_csv(OUT_DIR / "edges.csv", index=False)
(top_shared).to_csv(OUT_DIR / "top_shared_addresses.csv", index=False)
print(f"Wrote: {OUT_DIR/'nodes.csv'} , {OUT_DIR/'edges.csv'} and {OUT_DIR / 'top_shared_addresses.csv'}")
