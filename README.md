# LLM Pricing Calculator

The **LLM Pricing Calculator** is a modular Python tool that fetches pricing data for large language models (LLMs) from multiple online sources, stores the data in a SQLite database, and calculates the cost of processing a given text prompt based on token counts. The tool supports both interactive command-line usage and module-based integration into other Python programs.

## Features

- **Multi-Source Pricing Fetching:**  
  Scrapes pricing data from various websites such as:
  - [llm-price.today](https://llm-price.today/)
  - [llmpricecheck.com](https://llmpricecheck.com/calculator/)
  - [yourgpt.ai](https://yourgpt.ai/tools/openai-and-other-llm-api-pricing-calculator)

- **Data Normalization & Persistence:**  
  Normalizes all pricing values to a per 1,000,000 tokens basis and stores them in an SQLite database for persistent caching.

- **Interactive & CLI Support:**  
  Supports command-line arguments for direct usage, and interactive prompts for model selection (with fuzzy matching) when no command-line arguments are provided.

- **Fuzzy Matching for Model Names:**  
  Performs case-insensitive matching to suggest correct model names if the provided input does not exactly match any stored model name.

- **Cost Calculation:**  
  Uses a Hugging Face tokenizer (default: GPT-2) to compute token counts for a given prompt and calculates the cost based on the stored pricing data.

- **Modular Design:**  
  The tool can be imported as a module into other Python projects to integrate LLM pricing and cost calculation functionality.

## Requirements

- Python 3.7+
- Required Python libraries:
  - `requests`
  - `beautifulsoup4`
  - `sqlite3` (standard library)
  - `transformers`
  - `urllib3`
  - `argparse` (standard library)
  - `difflib` (standard library)

You can install the non-standard dependencies using pip:

```bash
pip install requests beautifulsoup4 transformers urllib3
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/vinayvinu500/llm-pricing-calculator.git
   cd llm-pricing-calculator
   ```

2. **Install Dependencies:**

   If you have a `requirements.txt` file (optional), run:

   ```bash
   pip install -r requirements.txt
   ```

   Or install the dependencies manually as shown above.

## Usage

### Command-Line Interface

You can run the tool directly from the command line:

- **Fetch Latest Pricing Data:**

  Force a fresh fetch from the online sources and update the database:

  ```bash
  python llm_pricing.py --fetch
  ```

- **List Existing Pricing Data:**

  List all stored pricing records from the SQLite database:

  ```bash
  python llm_pricing.py --list
  ```

- **Calculate Cost:**

  To calculate the cost for a given model and prompt, provide the `--model` and `--prompt` arguments:

  ```bash
  python llm_pricing.py --model "gpt-4o-mini" --prompt "Hello World"
  ```

  If you do not provide these arguments, the script will interactively ask for them. If the model name does not match exactly, fuzzy matching will be used to suggest close matches. If no close matches are found, the user is given an option to force a data refresh.

### Module Integration

You can also import the functions into your own Python scripts:

```python
from llm_pricing import init_db, get_pricing_data, calculate_cost

# Initialize the database (if not already done)
init_db()

# Load pricing data (use force_fetch=True to force a fresh scrape)
get_pricing_data(force_fetch=False)

# Calculate cost for a given prompt and model name
result = calculate_cost("Hello World", "gpt-4o-mini", token_type="input")
print(result)
```

## Code Structure

- **Database Functions:**  
  Functions to initialize the SQLite database, update the schema, load data, and update records.

- **Fetching Functions:**  
  Functions to scrape pricing data from different online sources and merge them into a unified in-memory store and persistent database.

- **Cost Calculation:**  
  Uses Hugging Face’s `AutoTokenizer` (configured for GPT-2 by default) to calculate the number of tokens, words, and characters in the prompt, then computes the cost based on the model's pricing.

- **Interactive CLI:**  
  The `main()` function handles command-line arguments and interactive prompts, including fuzzy matching for model selection.

## Future Improvements

- **Enhanced Logging:**  
  Integrate the Python `logging` module for more robust logging and debugging.

- **Unit Testing:**  
  Add unit tests (using `unittest` or `pytest`) to ensure individual functions behave as expected.

- **Configuration File:**  
  Allow configuration parameters (such as API endpoints or database file paths) to be loaded from an external configuration file.

## Limitations

- **Scraping Dependency:**  
  The tool depends on the structure of the target websites. Changes in their HTML layout may require updates to the scraping logic.

- **Tokenization Model:**  
  The default tokenizer is set to GPT-2. You may need to adjust or configure this if you require tokenization compatible with other models.

## License

This project is open source and available under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you find bugs or have suggestions for improvements.

---

Feel free to adjust or expand this README as needed for your project!