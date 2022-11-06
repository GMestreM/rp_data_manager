# Data manager for Risk Parity Dashboard

This service retrieves financial data from a data provider and stores it into a SQL database. Once the data is stored in the database, a systematic portfolio construction model is executed and its output stored in the database. The systematic investing strategy is based on a [Risk Parity](https://www.investopedia.com/terms/r/risk-parity.asp) model, which aims at obtaining optimized diversification across all assets by allocating capital on a risk-weighted basis, so they contribute equally to the risk of our portfolio. 

# Database schema

Both raw financial data and allocation weights are stored in an cloud-hosted SQL database. This service expects the following tables in the production database:

* `asset_header`: contains information about each asset used by the models. All assets contained in this table will be queried and included in table `asset_prices`

	| `id`      | `isin`        | `name`    | `currency`    | `country` |  
	| -----     | ----------    | --------  | --------      | --------  |
	| (index)   |               |           |               |           |
	|   _int_   |  _string_     | _string_  | _string_      | _string_  |

* `asset_prices`:
	* multi-index: `id_asset`, `date` and `timestamp`.
	* Data values are returned from [Investing](https://investing.com)
	
	| `id_asset`                | `date`    | `timestamp`   | `open_price`  | `high_price`  |  `low_price`  | `close_price` | `volume`  | `change_pct`  |   
	| ----                      | ----      | ----          | ----          | ----          | ----          | ----          | ----      | ----          |
	| (index)                   | (index)   | (index)       |               |               |               |               |           |               |
	| _int_ (`asset_header.id`) | _date_    |_date_         | _float_       | _float_       |_float_        |_float_        |_int_      | _float_       |     
	
* `model_header`:
	* only one row can have its `flag` set to 1. This indicates the active model.
	* `assets` will contain an string such as `1;3;4`, with each `asset_header.id` used by the model separated by commas.
	* `constr_dict` contains a string specifying constraints included in the model, using a json-like format (i.e. `{{id_asset_1;id_asset_2 < constraint}}` stands for sum of weight of asset `id_asset_1` and `id_asset_2` will be lower than `constraint`)

	| `id`      | `flag`    | `assets`  | `window_size` | `constr_dict` | `rp_model`    | `risk_measure`    |
	| ----      | ----      | ----      | ----          | ----          |----           | ----              |
	| (index)   |           |           |               |               |               |                   |
	| _int_     | _bool_    | _str_     | _int_         | _str_         | _str_         | _str_             |
	
* `model_weights`:
	* multi-index: `id_model`,`id_asset`,`timestamp` and `date`.

	|`id_model`                 |`id_asset`                 |`timestamp`|`date`     |`weight`   |
	| ----                      | ----                      | ----      | ----      | ----      |
	| (index)                   | (index)                   | (index)   | (index)   |           |
	| _int_(`asset_header.id`)  | _int_ (`model_header.id`) | _date_    |_date_     |_float_    |

### Environment variables
- `DB_USER_NAME`
- `DB_USER_PWD`
- `DB_URL_PATH`
- `DB_URL_PORT`