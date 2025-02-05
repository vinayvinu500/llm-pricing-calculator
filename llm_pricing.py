#!/usr/bin/env python3
"""
Module: llm_pricing
Description: Provides functions to fetch pricing data from multiple sources,
calculate cost for given text using stored pricing, and list existing pricing data.
Can be used both as an importable module and as a standalone CLI tool.
"""

import requests
from bs4 import BeautifulSoup
import sqlite3
import datetime
from transformers import AutoTokenizer
import urllib3
import argparse
import sys
import difflib

# Disable warnings for insecure SSL (for development only)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Global in-memory key–value store to hold pricing data.
# All pricing is normalized to be per 1M tokens.
price_store = {}

# SQLite DB filename
DB_FILE = 'prices.db'

# -------------------- Database and Schema Functions --------------------

def init_db():
    """Initialize the SQLite database and create the prices table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Create table with default values for the source columns.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            model_name TEXT PRIMARY KEY,
            provider TEXT,
            input_price REAL,
            output_price REAL,
            updated_at TEXT,
            input_reference TEXT DEFAULT 'unknown',
            output_reference TEXT DEFAULT 'unknown'
        )
    ''')
    conn.commit()
    conn.close()
    update_schema()  # Ensure that the table has all required columns

def update_schema():
    """
    Update the schema of the prices table by adding new columns if they don't exist.
    Specifically, add input_reference and output_reference.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(prices)")
    columns = [col[1] for col in cursor.fetchall()]
    if "input_reference" not in columns:
        cursor.execute("ALTER TABLE prices ADD COLUMN input_reference TEXT DEFAULT 'unknown'")
    if "output_reference" not in columns:
        cursor.execute("ALTER TABLE prices ADD COLUMN output_reference TEXT DEFAULT 'unknown'")
    conn.commit()
    conn.close()

def update_db(model_name, provider, input_price, output_price, input_reference, output_reference):
    """Insert or update a model's pricing details in the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    updated_at = datetime.datetime.now().isoformat()
    # Ensure that the reference values are never None.
    input_reference = input_reference if input_reference is not None else "unknown"
    output_reference = output_reference if output_reference is not None else "unknown"
    cursor.execute('''
        INSERT INTO prices (model_name, provider, input_price, output_price, updated_at, input_reference, output_reference)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_name) DO UPDATE SET 
            provider=excluded.provider,
            input_price=excluded.input_price,
            output_price=excluded.output_price,
            updated_at=excluded.updated_at,
            input_reference=excluded.input_reference,
            output_reference=excluded.output_reference
    ''', (model_name, provider, input_price, output_price, updated_at, input_reference, output_reference))
    conn.commit()
    conn.close()

def load_db_data():
    """
    Load pricing data from the SQLite database into the global price_store.
    Replace any missing source values with a default.
    """
    global price_store
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT model_name, provider, input_price, output_price, input_reference, output_reference, updated_at FROM prices")
    rows = cursor.fetchall()
    for row in rows:
        model_name, provider, input_price, output_price, input_reference, output_reference, updated_at = row
        # Replace None with default if needed.
        input_reference = input_reference if input_reference is not None else "unknown"
        output_reference = output_reference if output_reference is not None else "unknown"
        price_store[model_name] = {
            "provider": provider,
            "input_price": input_price,
            "output_price": output_price,
            "input_reference": input_reference,
            "output_reference": output_reference,
            "updated_at": updated_at
        }
    conn.close()

def get_last_fetched_date():
    """
    Get the most recent pricing update date from the price_store.
    """
    if not price_store:
        return None
    dates = [datetime.datetime.fromisoformat(info["updated_at"]) for info in price_store.values() if info.get("updated_at")]
    return max(dates).isoformat() if dates else None

# -------------------- Fetching Functions --------------------

def merge_pricing(new_data: dict):
    """
    Merge new pricing data into the global price_store and update the database
    only if the new input or output prices are higher.
    """
    global price_store
    for model, new_info in new_data.items():
        if model not in price_store:
            price_store[model] = new_info
            update_db(model, new_info["provider"], new_info["input_price"], new_info["output_price"],
                      new_info["input_reference"], new_info["output_reference"])
        else:
            current = price_store[model]
            updated = False
            if new_info["input_price"] > current["input_price"]:
                current["input_price"] = new_info["input_price"]
                current["input_reference"] = new_info["input_reference"]
                updated = True
            if new_info["output_price"] > current["output_price"]:
                current["output_price"] = new_info["output_price"]
                current["output_reference"] = new_info["output_reference"]
                updated = True
            if updated:
                current["provider"] = new_info["provider"]
                current["updated_at"] = datetime.datetime.now().isoformat()
                update_db(model, current["provider"], current["input_price"], current["output_price"],
                          current["input_reference"], current["output_reference"])
    return price_store

def fetch_from_llm_price_today():
    """
    Scrape https://llm-price.today/ which has columns:
      "Model Name", "Provider", "Input Price / 1M Tokens", "Output Price / 1M Tokens"
    """
    url = "https://llm-price.today/"
    new_data = {}
    try:
        response = requests.get(url, verify=False)
        if response.status_code != 200:
            print(f"Failed to fetch {url}: status code {response.status_code}")
            return new_data
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find("table")
        if not table:
            print("No table found on llm-price.today")
            return new_data
        rows = table.find_all("tr")
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) >= 4:
                model_name = cols[0].text.strip()
                provider = cols[1].text.strip()
                try:
                    input_price = float(cols[2].text.strip().replace("$", ""))
                    output_price = float(cols[3].text.strip().replace("$", ""))
                except ValueError:
                    continue
                new_data[model_name] = {
                    "provider": provider,
                    "input_price": input_price,
                    "output_price": output_price,
                    "input_reference": "llm-price.today",
                    "output_reference": "llm-price.today",
                    "updated_at": datetime.datetime.now().isoformat()
                }
    except Exception as e:
        print(f"Error fetching from llm-price.today: {e}")
    return new_data

def fetch_from_llmpricecheck():
    """
    Scrape https://llmpricecheck.com/calculator/ which has columns:
      "Model", "Provider", "Context", "Input $/1M", "Output $/1M", "Per Call", "Total"
    """
    url = "https://llmpricecheck.com/calculator/"
    new_data = {}
    try:
        response = requests.get(url, verify=False)
        if response.status_code != 200:
            print(f"Failed to fetch {url}: status code {response.status_code}")
            return new_data
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find("table")
        if not table:
            print("No table found on llmpricecheck.com")
            return new_data
        rows = table.find_all("tr")
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) >= 5:
                model_name = cols[0].text.strip()
                provider = cols[1].text.strip()
                try:
                    input_price = float(cols[3].text.strip().replace("$", ""))
                    output_price = float(cols[4].text.strip().replace("$", ""))
                except ValueError:
                    continue
                new_data[model_name] = {
                    "provider": provider,
                    "input_price": input_price,
                    "output_price": output_price,
                    "input_reference": "llmpricecheck.com",
                    "output_reference": "llmpricecheck.com",
                    "updated_at": datetime.datetime.now().isoformat()
                }
    except Exception as e:
        print(f"Error fetching from llmpricecheck.com: {e}")
    return new_data

def fetch_from_yourgpt():
    """
    Scrape https://yourgpt.ai/tools/openai-and-other-llm-api-pricing-calculator which has columns:
      "Provider", "Model", "Context", "Input/1k Tokens", "Output/1k Tokens", "Per Call", "Total"
    Convert prices per 1k tokens to per 1M tokens by multiplying by 1000.
    """
    url = "https://yourgpt.ai/tools/openai-and-other-llm-api-pricing-calculator"
    new_data = {}
    try:
        response = requests.get(url, verify=False)
        if response.status_code != 200:
            print(f"Failed to fetch {url}: status code {response.status_code}")
            return new_data
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find("table")
        if not table:
            print("No table found on yourgpt.ai")
            return new_data
        rows = table.find_all("tr")
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) >= 5:
                provider = cols[0].text.strip()
                model_name = cols[1].text.strip()
                try:
                    input_price = float(cols[3].text.strip().replace("$", "")) * 1000
                    output_price = float(cols[4].text.strip().replace("$", "")) * 1000
                except ValueError:
                    continue
                new_data[model_name] = {
                    "provider": provider,
                    "input_price": input_price,
                    "output_price": output_price,
                    "input_reference": "yourgpt.ai",
                    "output_reference": "yourgpt.ai",
                    "updated_at": datetime.datetime.now().isoformat()
                }
    except Exception as e:
        print(f"Error fetching from yourgpt.ai: {e}")
    return new_data

def fetch_latest_prices():
    """
    Aggregate pricing information from multiple sources.
    Merge pricing if new values are higher than the existing ones.
    """
    all_new_data = {}
    sources = [
        fetch_from_llm_price_today,
        fetch_from_llmpricecheck,
        fetch_from_yourgpt
    ]
    
    for source in sources:
        data = source()
        for model, info in data.items():
            if model in all_new_data:
                existing = all_new_data[model]
                if info["input_price"] > existing["input_price"]:
                    existing["input_price"] = info["input_price"]
                    existing["input_reference"] = info["input_reference"]
                if info["output_price"] > existing["output_price"]:
                    existing["output_price"] = info["output_price"]
                    existing["output_reference"] = info["output_reference"]
                existing["provider"] = info["provider"]
                existing["updated_at"] = datetime.datetime.now().isoformat()
            else:
                all_new_data[model] = info
    merge_pricing(all_new_data)
    return price_store

def get_pricing_data(force_fetch: bool = False):
    """
    Return pricing data. If force_fetch is True, scrape fresh data; otherwise,
    load from the in-memory store or the database.
    """
    global price_store
    if force_fetch:
        fetch_latest_prices()
    else:
        if not price_store:
            load_db_data()
    return price_store

def list_db_data():
    """
    List all pricing records currently stored in the database.
    """
    load_db_data()
    return price_store

# -------------------- Cost Calculation and Tokenization --------------------

# Initialize the tokenizer (using GPT-2 as an example)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def calculate_cost(text: str, model_name: str, token_type: str = "input"):
    """
    Given a text and a model name, calculate token, word, and character counts,
    then compute the cost based on the model's pricing.
    Returns a dictionary with counts, cost, pricing reference, and last pricing update date.
    """
    if model_name not in price_store:
        raise ValueError(f"Model '{model_name}' not found in the pricing store. Please fetch the latest prices first.")
    
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    char_count = len(text)
    word_count = len(text.split())
    
    model_info = price_store[model_name]
    if token_type == "input":
        price_per_million = model_info["input_price"]
    elif token_type == "output":
        price_per_million = model_info["output_price"]
    else:
        raise ValueError("token_type must be either 'input' or 'output'")
    
    cost = (token_count / 1_000_000) * price_per_million
    last_pricing_date = get_last_fetched_date()
    
    return {
        "model_name": model_name,
        "provider": model_info["provider"],
        "token_count": token_count,
        "word_count": word_count,
        "char_count": char_count,
        "cost": cost,
        "last_pricing_date": last_pricing_date,
        "pricing_reference": {
            "input": {"price": model_info["input_price"], "source": model_info["input_reference"]},
            "output": {"price": model_info["output_price"], "source": model_info["output_reference"]}
        }
    }

# -------------------- Fuzzy Matching Helper --------------------
def find_model_case_insensitive(model_input: str):
    """
    Check if a model exists in price_store, ignoring case.
    If found, return the actual key from price_store; otherwise, return None.
    """
    for key in price_store.keys():
        if key.lower() == model_input.lower():
            return key
    return None

def suggest_model(model_input: str):
    """
    Return a list of model names that closely match the user input,
    using case-insensitive matching.
    """
    lower_input = model_input.lower()
    # Create a mapping from lower-case model names to their original versions.
    lower_keys = {k.lower(): k for k in price_store.keys()}
    # Get close matches using the lower-case keys.
    matches = difflib.get_close_matches(lower_input, lower_keys.keys(), n=5, cutoff=0.3)
    # Return the original model names corresponding to the lower-case matches.
    return [lower_keys[m] for m in matches]


# -------------------- Command-line Interface --------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fetch LLM pricing data and calculate cost for a given text prompt."
    )
    parser.add_argument("--fetch", action="store_true",
                        help="Force fetch the latest pricing data from the web (otherwise, load from DB).")
    parser.add_argument("--list", action="store_true",
                        help="List all pricing data currently stored in the database.")
    parser.add_argument("--model", type=str, help="Model name for cost calculation.")
    parser.add_argument("--prompt", type=str, help="Input text prompt for which to calculate cost.")
    parser.add_argument("--token-type", type=str, choices=["input", "output"], default="input",
                        help="Specify token type: 'input' (default) or 'output'.")
    
    args = parser.parse_args()

    # Initialize DB and update schema
    init_db()

    # If --list is provided, list all pricing data.
    if args.list:
        data = list_db_data()
        if data:
            for model, info in data.items():
                print(f"{model}: {info}")
        else:
            print("No pricing data found in the database.")
        sys.exit(0)

    # Get pricing data (either fetch new data or load from DB)
    get_pricing_data(force_fetch=args.fetch)

    # If model and prompt are not provided via CLI, ask interactively.
    if not args.model:
        args.model = input("Enter model name: ").strip()
    if not args.prompt:
        args.prompt = input("Enter prompt text: ").strip()

    # Try to find an exact match ignoring case.
    matched_model = find_model_case_insensitive(args.model)
    if matched_model:
        args.model = matched_model
    else:
        # Fuzzy matching: if the model is not found, suggest alternatives.
        suggestions = suggest_model(args.model)
        if not suggestions:
            # No suggestions found; ask if user wants to fetch the latest pricing data.
            user_choice = input("No similar models found. Would you like to fetch the latest pricing data? (y/n): ").strip().lower()
            if user_choice == 'y':
                get_pricing_data(force_fetch=True)
                suggestions = suggest_model(args.model)
            else:
                print("Exiting. Please re-check your input or fetch the latest pricing data separately.")
                sys.exit(1)
        if suggestions:
            print(f"Model '{args.model}' not found. Did you mean:")
            for i, model in enumerate(suggestions, start=1):
                print(f"  {i}. {model}")
            selection = input("Enter the number of the correct model (or press Enter to cancel): ").strip()
            if selection.isdigit():
                idx = int(selection) - 1
                if 0 <= idx < len(suggestions):
                    args.model = suggestions[idx]
                else:
                    print("Invalid selection. Exiting.")
                    sys.exit(1)
            else:
                print("No valid selection made. Exiting.")
                sys.exit(1)

    # Calculate cost using the finalized model name and prompt.
    try:
        result = calculate_cost(args.prompt, args.model, token_type=args.token_type)
        print("Cost Calculation Result:")
        print(result)
    except Exception as e:
        print(f"Error in cost calculation: {e}")

# -------------------- Module Interface --------------------
# These functions can be imported and used in other Python scripts.
__all__ = [
    "init_db", "get_pricing_data", "list_db_data", "calculate_cost",
    "fetch_latest_prices", "load_db_data", "suggest_model"
]

if __name__ == "__main__":
    main()

"""
# Usage
Force Fetch Pricing Data: To scrape fresh pricing data and update the database
`python llm_pricing.py --fetch --list`

Calculate Cost for a Given Prompt: Once pricing data is available in the database, you can calculate the cost
`python llm_pricing.py --model "gpt-4o-mini" --prompt "Hello World"`

List Existing Pricing Data: To list all records from the database
`python llm_pricing.py --list`

# Function Calls
This function lets you decide whether to scrape fresh data or to load existing pricing from the database/in-memory store.
`get_pricing_data(force_fetch=False)`

Loads existing data from SQLite into price_store if it’s empty.
`load_db_data()`

Computes the most recent update date among all pricing records so that the calculate_cost() output includes the timestamp of the last pricing update.
`get_last_fetched_date()`

Function returns a JSON structure with the token, word, and character counts, the calculated cost, the pricing references (with both input and output details), and the last pricing update date.
The calculate_cost() 

------------------------------------------------
# Sample 1
> python llm_pricing.py
Enter model name: GPT-4o mini
Enter prompt text: Hello World
No similar models found. Would you like to fetch the latest pricing data? (y/n): y
Model 'GPT-4o mini' not found. Did you mean:
  1. GPT-4o mini
  2. GPT-4o (omni)
  3. GPT-4o mini Realtime
  4. gpt-4o-mini
  5. GPT-4
Enter the number of the correct model (or press Enter to cancel): 1
Cost Calculation Result:
{'model_name': 'GPT-4o mini', 'provider': 'OpenAI', 'token_count': 2, 'word_count': 2, 'char_count': 11, 'cost': 3e-07, 'last_pricing_date': '2025-02-05T21:15:04.304099', 'pricing_reference': {'input': {'price': 0.15, 'source': 'yourgpt.ai'}, 'output': {'price': 0.6, 'source': 'yourgpt.ai'}}}

# Sample 2
> python llm_pricing.py
Enter model name: deepseek-r1
Enter prompt text: Hello World
Cost Calculation Result:
{'model_name': 'DeepSeek-R1', 'provider': 'DeepSeek', 'token_count': 2, 'word_count': 2, 'char_count': 11, 'cost': 1.1e-06, 'last_pricing_date': '2025-02-05T21:15:05.107267', 'pricing_reference': {'input': {'price': 0.55, 'source': 'yourgpt.ai'}, 'output': {'price': 2.19, 'source': 'yourgpt.ai'}}}

# Sample 3
python llm_pricing.py --model "gpt-4o-mini" --prompt "Hello World"
Cost Calculation Result:
{'model_name': 'gpt-4o-mini', 'provider': 'OpenAI', 'token_count': 2, 'word_count': 2, 'char_count': 11, 'cost': 3e-07, 'last_pricing_date': '2025-02-05T21:15:05.107267', 'pricing_reference': {'input': {'price': 0.15, 'source': 'llm-price.today'}, 'output': {'price': 0.6, 'source': 'llm-price.today'}}}

# Bad Entry
Enter model name: GPT-4 mini
Enter prompt text: Hello World
No similar models found. Please fetch the latest pricing data or re-check your input.
"""