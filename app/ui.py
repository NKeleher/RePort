import json
from io import StringIO

import pandas as pd
import streamlit as st
from database import DuckDB, get_query_string

db = DuckDB()


class Sidebar:
    """Class to handle the sidebar components of the Streamlit app."""

    def header():
        """Display the header for the sidebar."""
        st.header("Configurations")

    def radio():
        """Display a radio button for input method selection.

        Returns
        -------
        str
            The selected input method.

        """
        input_method = st.radio("Input Method", ("File", "Manual"))
        return input_method

    def select_box():
        """Display a select box for rebalance type selection.

        Returns
        -------
        str
            The selected rebalance type.

        """
        rebalance_type = st.selectbox(
            "Select Rebalance Type",
            ("Investable Cash Dynamic", "Investable Cash Target", "Whole Portfolio"),
        )
        return rebalance_type

    def check_box_frac_shares():
        """Display a checkbox for fractional share investing option.

        Returns
        -------
        bool
            True if fractional share investing is allowed, False otherwise.

        """
        is_frac_shares = st.checkbox("Allow fractional share investing?")
        return is_frac_shares

    def check_box_sample_data():
        """Display a checkbox for adding sample data.

        Returns
        -------
        bool
            True if sample data is to be added, False otherwise.

        """

        def add_sample_data():
            if st.session_state.add_sample_data:
                db.query("DELETE FROM cash")
                db.query("DELETE FROM holdings")
                db.query(
                    """
                INSERT INTO
                    cash
                SELECT
                    *
                FROM
                    read_csv_auto('app/data/example_cash.csv')
                """
                )
                db.query(
                    """
                INSERT INTO
                    holdings
                SELECT
                    md5(concat(lower(trim(account_name)),lower(trim(ticker)))),
                    *
                FROM
                    read_csv_auto('app/data/example_holdings.csv')
                """
                )
            else:
                db.query("DELETE FROM cash")
                db.query("DELETE FROM holdings")

        add_sample_data = st.checkbox(
            "Add sample data?",
            help="Warning: This will delete all current data!",
            on_change=add_sample_data,
            key="add_sample_data",
        )

        return add_sample_data

    def select_brokerage_platform():
        """Display a select box for brokerage platform selection.

        Returns
        -------
        str
            The selected brokerage platform.

        """
        brokerage_platform = st.selectbox(
            "Select Brokerage Platform",
            ("Charles Schwab",),
        )
        return brokerage_platform

    def file_upload_holdings():
        """Display a file uploader for holdings file.

        Returns
        -------
        StringIO or None
            The contents of the uploaded file, or None if no file is uploaded.

        """
        holdings_file = st.file_uploader(
            "Upload Holdings file", accept_multiple_files=False
        )
        file_contents = None
        if holdings_file is not None:
            file_contents = StringIO(holdings_file.getvalue().decode("utf-8"))
        return file_contents

    def file_upload_target_weights():
        """Display a file uploader for target weights JSON file.

        Returns
        -------
        dict or None
            The contents of the uploaded JSON file, or None if no file is uploaded.

        """
        target_weights_file = st.file_uploader(
            "Upload target weights json file", accept_multiple_files=False
        )
        file_contents = None
        if target_weights_file is not None:
            file_contents = json.loads(target_weights_file.getvalue())
        return file_contents

    def check_box_holdings_data(brokerage_platform, holdings_file, target_weights):
        """Display a checkbox for adding holdings data.

        Parameters
        ----------
        brokerage_platform : str
            The selected brokerage platform.
        holdings_file : StringIO
            The contents of the uploaded holdings file.
        target_weights : dict
            The target weights data.

        Returns
        -------
        bool
            True if holdings data is to be added, False otherwise.

        """

        def add_holdings_data(brokerage_platform, holdings_file, target_weights):
            if st.session_state.add_holdings_data:
                db.query("DELETE FROM cash")
                db.query("DELETE FROM holdings")

                holdings_df = None
                if brokerage_platform == "Charles Schwab":
                    holdings_df = charles_schwab_file_parser(  # noqa: F841
                        holdings_file, target_weights
                    )
                else:
                    st.error("We don't currently support that brokerage platform")

                db.query(
                    """
                INSERT INTO
                    cash
                SELECT
                    account_name,
                    shares as cash
                FROM
                    holdings_df
                WHERE
                    ticker = 'Cash & Cash Investments'
                """
                )
                db.query(
                    """
                INSERT INTO
                    holdings
                SELECT
                    md5(concat(lower(trim(account_name)),lower(trim(ticker)))),
                    *
                FROM
                    holdings_df
                WHERE
                    ticker != 'Cash & Cash Investments'
                """
                )
            else:
                db.query("DELETE FROM cash")
                db.query("DELETE FROM holdings")

        def charles_schwab_file_parser(file, target_weights):
            """Parse the Charles Schwab holdings file.

            Parameters
            ----------
            file : StringIO
                The contents of the uploaded holdings file.
            target_weights : dict
                The target weights data.

            Returns
            -------
            pd.DataFrame
                The parsed holdings data as a DataFrame.

            """
            holdings_dict = {
                "account_name": [],
                "ticker": [],
                "security_name": [],
                "shares": [],
                "target_weight": [],
                "cost": [],
                "price": [],
            }
            # Skip the first two lines
            file.readline()
            file.readline()

            accounts = file.read().split(
                '\n"","","","","","","","","","","","","","","","",""\n"","","","","","","","","","","","","","","","",""\n'
            )

            for account in accounts:
                lines = [line.strip() for line in account.split("\n")]
                # first line is account
                account_name = (
                    lines[0].split(",")[0].split()[0].replace("_", " ").replace('"', "")
                )
                # second line is headers
                headers = [header.replace('"', "") for header in lines[1].split(",")]  # noqa: F841
                # the rest are the actual values
                for symbol in lines[2:]:
                    values = [value.replace('"', "") for value in symbol.split(",")]
                    if values[0] == "Account Total" or values[0] == "":
                        continue
                    elif values[0] == "Cash & Cash Investments":
                        holdings_dict["account_name"].append(account_name)
                        holdings_dict["ticker"].append(values[0])
                        holdings_dict["security_name"].append(values[1])
                        holdings_dict["shares"].append(values[6].replace("$", ""))
                        holdings_dict["target_weight"].append(0)
                        holdings_dict["cost"].append(values[9].replace("$", ""))
                        holdings_dict["price"].append(values[3].replace("$", ""))
                    else:
                        holdings_dict["account_name"].append(account_name)
                        holdings_dict["ticker"].append(values[0])
                        holdings_dict["security_name"].append(values[1])
                        holdings_dict["shares"].append(values[2])
                        holdings_dict["target_weight"].append(
                            target_weights[account_name][values[0]]
                        )
                        holdings_dict["cost"].append(values[9].replace("$", ""))
                        holdings_dict["price"].append(values[3].replace("$", ""))
            return pd.DataFrame(holdings_dict)

        add_holdings_data_checkbox = st.checkbox(
            "Add holdings data?",
            help="Warning: This will delete all current data!",
            on_change=lambda: add_holdings_data(
                brokerage_platform, holdings_file, target_weights
            ),
            key="add_holdings_data",
        )

        return add_holdings_data_checkbox


class CashInput:
    """Class to handle the cash input components of the Streamlit app."""

    def file():
        """Display a file uploader for cash file.

        Returns
        -------
        None

        """
        uploaded_data = st.file_uploader(
            "Drag and Drop Cash File or Click to Upload",
            type=".csv",
            accept_multiple_files=False,
        )

        if uploaded_data is not None:
            st.success("Uploaded your file!")
            uploaded_data = uploaded_data

            df = pd.read_csv(uploaded_data)  # noqa: F841

            db.query("DELETE FROM cash")
            db.query("INSERT INTO cash SELECT * FROM df")
        return None


class HoldingsInput:
    """Class to handle the holdings input components of the Streamlit app."""

    def file():
        """Display a file uploader for holdings file.

        Returns
        -------
        None

        """
        uploaded_data = st.file_uploader(
            "Drag and Drop Holdings File or Click to Upload",
            type=".csv",
            accept_multiple_files=False,
        )
        if uploaded_data is not None:
            st.success("Uploaded your file!")
            uploaded_data = uploaded_data

            df = pd.read_csv(uploaded_data)  # noqa: F841

            db.query("DELETE FROM holdings")
            db.query(
                """
            INSERT INTO
                holdings
            SELECT
                md5(concat(lower(trim(account_name)),lower(trim(ticker)))),
                *
            FROM df
            """
            )
        return None


class Portfolio:
    """Class to handle the portfolio operations of the Streamlit app."""

    def create_tables(rebalance_type, is_frac_shares):
        """Create the necessary tables for the portfolio.

        Parameters
        ----------
        rebalance_type : str
            The type of rebalance to perform.
        is_frac_shares : bool
            Whether fractional shares are allowed.

        """
        db.query(get_query_string("create_holdings_table"))
        db.query(get_query_string("create_cash_table"))
        Portfolio.create_future_holdings(rebalance_type, is_frac_shares)

    def update_tables(table: tuple, df: pd.DataFrame) -> None:
        """Update the specified table with the given DataFrame.

        Parameters
        ----------
        table : tuple
            The table to update.
        df : pd.DataFrame
            The DataFrame containing the new data.

        """
        if table == "cash":
            db.query("DELETE FROM cash")
            db.query(
                """
                INSERT INTO
                    cash
                SELECT
                    *
                FROM
                    df
                """
            )
        elif table == "holdings":
            db.query("DELETE FROM holdings")
            db.query(
                """
            INSERT INTO
                holdings
            SELECT
                md5(concat(lower(trim(account_name)),lower(trim(ticker)))),
                *
            FROM
                df
            """
            )
        return

    def get_raw_holdings_table():
        """Fetch the raw holdings table.

        Returns
        -------
        pd.DataFrame
            The raw holdings table.

        """
        return db.fetch(get_query_string("select_raw_holdings"))

    def get_raw_cash_table():
        """Fetch the raw cash table.

        Returns
        -------
        pd.DataFrame
            The raw cash table.

        """
        return db.fetch(get_query_string("select_raw_cash"))

    def get_accounts():
        """Fetch the list of accounts.

        Returns
        -------
        list
            The list of accounts.

        """
        accounts = list(
            map(
                lambda _tup: str(_tup[0]),
                db.fetch("SELECT DISTINCT account_name FROM cash", return_df=False),
            )
        )
        return accounts

    def dynamic_invest(account):
        """Perform dynamic investment for the given account.

        Parameters
        ----------
        account : str
            The account to perform dynamic investment for.

        """
        df = db.fetch(get_query_string("select_future_holdings"))

        cash = db.fetch(
            f"SELECT cash FROM future_cash WHERE account_name = '{account}'",
            return_df=False,
        )[0][0]

        is_cash_left = bool(
            db.fetch(
                f"""
            SELECT
                *
            FROM
                df
            WHERE
                price <= {cash}
                AND account_name = '{account}'
            """,
                return_df=False,
            )
        )

        while is_cash_left:
            ticker, shares, new_cash, cost = db.fetch(
                f"""
                SELECT
                    ticker,
                    1 + shares,
                    cash - price as new_cash,
                    cost + price as cost
                FROM
                    df
                WHERE
                    price <= {cash}
                    AND account_name = '{account}'
                ORDER BY
                    target_diff
                Limit 1
                """,
                return_df=False,
            )[0]

            db.query(
                """
                UPDATE future_holdings SET
                    shares = ?,
                    cost = ?
                WHERE
                    account_name    = ?
                AND ticker          = ?
                """,
                [(shares, cost, account, ticker)],
            )

            db.query(
                """
                UPDATE future_cash SET
                    cash = ?
                WHERE
                    account_name    = ?
                """,
                [(new_cash, account)],
            )

            df = db.fetch(get_query_string("select_future_holdings"))  # noqa: F841

            cash = db.fetch(
                f"SELECT cash FROM future_cash WHERE account_name = '{account}'",
                return_df=False,
            )[0][0]

            is_cash_left = bool(
                db.fetch(
                    f"""
                SELECT
                    *
                FROM
                    df
                WHERE
                    price <= {cash}
                    AND account_name = '{account}'
                """,
                    return_df=False,
                )
            )

        return

    def create_future_holdings(rebalance_type, is_frac_shares):
        """Create the future holdings table based on the rebalance type and fractional shares setting.

        Parameters
        ----------
        rebalance_type : str
            The type of rebalance to perform.
        is_frac_shares : bool
            Whether fractional shares are allowed.

        """
        df = db.fetch(get_query_string("select_holdings"))  # noqa: F841

        db.fetch("DROP TABLE IF EXISTS future_holdings")
        db.fetch("DROP TABLE IF EXISTS future_cash")

        if rebalance_type == "Investable Cash Dynamic" and is_frac_shares:
            column = "dynamic_shares_to_invest_frac"
        elif rebalance_type == "Investable Cash Dynamic" and not is_frac_shares:
            column = "dynamic_shares_to_invest_whole"
        elif rebalance_type == "Investable Cash Target" and is_frac_shares:
            column = "target_shares_to_invest_frac"
        elif rebalance_type == "Investable Cash Target" and not is_frac_shares:
            column = "target_shares_to_invest_whole"
        elif rebalance_type == "Whole Portfolio" and is_frac_shares:
            column = "all_shares_to_invest_frac"
        elif rebalance_type == "Whole Portfolio" and not is_frac_shares:
            column = "all_shares_to_invest_whole"
        else:
            None

        sql = f"""
        CREATE TABLE future_holdings as (
            SELECT
                holding_id,
                account_name,
                ticker,
                security_name,
                shares + {column} as shares,
                target_weight,
                cost + ({column} * price) as cost,
                price
             FROM df
            )
       """

        db.fetch(sql)

        db.fetch(
            f"""
        CREATE TABLE future_cash as (
            SELECT
                account_name,
                max(cash) - sum({column} * price) as cash
             FROM df
             GROUP BY 1
            )
       """
        )

        if rebalance_type == "Investable Cash Dynamic" and not is_frac_shares:
            accounts = list(
                map(
                    lambda _tup: str(_tup[0]),
                    db.fetch("SELECT DISTINCT account_name FROM cash", return_df=False),
                )
            )
            for account in accounts:
                Portfolio.dynamic_invest(account)
        return

    def get_holdings_df_filtered(accounts, index):
        """Fetch the filtered holdings DataFrame for the given account.

        Parameters
        ----------
        accounts : list
            The list of accounts.
        index : int
            The index of the selected account.

        Returns
        -------
        tuple
            The holdings DataFrame and future holdings DataFrame.

        """
        raw_holdings_df = db.fetch(get_query_string("select_holdings"))  # noqa: F841
        raw_future_holdings_df = db.fetch(get_query_string("select_future_holdings"))  # noqa: F841
        holdings_df = db.fetch(
            f"""
            SELECT *
            FROM raw_holdings_df
            WHERE account_name = '{accounts[index]}'
            """
        )

        future_holdings_df = db.fetch(
            f"""
            SELECT *
            FROM raw_future_holdings_df
            WHERE account_name = '{accounts[index]}'
            """
        )
        return holdings_df, future_holdings_df

    def investable_cash_metric(accounts, index, column):
        """Display the investable cash metric for the given account.

        Parameters
        ----------
        accounts : list
            The list of accounts.
        index : int
            The index of the selected account.
        column : streamlit.delta_generator.DeltaGenerator
            The Streamlit column to display the metric in.

        """
        holdings_df, future_holdings_df = Portfolio.get_holdings_df_filtered(
            accounts, index
        )

        cash = db.fetch("SELECT max(cash) FROM holdings_df", return_df=False)[0][0]
        cash_formatted = f"${cash:,.2f}"
        column.metric("Investable Cash", cash_formatted)
        return

    def future_cash_metric(accounts, index, column):
        """Display the future cash metric for the given account.

        Parameters
        ----------
        accounts : list
            The list of accounts.
        index : int
            The index of the selected account.
        column : streamlit.delta_generator.DeltaGenerator
            The Streamlit column to display the metric in.

        """
        holdings_df, future_holdings_df = Portfolio.get_holdings_df_filtered(
            accounts, index
        )
        cash = db.fetch("SELECT max(cash) FROM holdings_df", return_df=False)[0][0]
        future_cash = db.fetch(
            "SELECT max(cash) FROM future_holdings_df", return_df=False
        )[0][0]
        future_cash_formatted = f"${future_cash:,.2f}"
        cash_delta = f"{future_cash - cash:,.2f}"
        column.metric(
            "Investable Cash After Rebalance", future_cash_formatted, delta=cash_delta
        )
        return None

    def market_value_metric(accounts, index, column):
        """Display the market value metric for the given account.

        Parameters
        ----------
        accounts : list
            The list of accounts.
        index : int
            The index of the selected account.
        column : streamlit.delta_generator.DeltaGenerator
            The Streamlit column to display the metric in.

        """
        holdings_df, future_holdings_df = Portfolio.get_holdings_df_filtered(
            accounts, index
        )

        market_value = db.fetch(
            "SELECT sum(market_value) FROM holdings_df", return_df=False
        )[0][0]
        market_value_formatted = f"${market_value:,.2f}"

        gain_loss = db.fetch(
            "SELECT sum(market_value) - sum(cost) FROM holdings_df", return_df=False
        )[0][0]
        gain_loss_formatted = f"{gain_loss:,.2f}"
        column.metric(
            "Account Market Value", market_value_formatted, delta=gain_loss_formatted
        )
        return None

    def gain_loss_metric(accounts, index, column):
        """Display the gain/loss metric for the given account.

        Parameters
        ----------
        accounts : list
            The list of accounts.
        index : int
            The index of the selected account.
        column : streamlit.delta_generator.DeltaGenerator
            The Streamlit column to display the metric in.

        """
        holdings_df, future_holdings_df = Portfolio.get_holdings_df_filtered(
            accounts, index
        )
        gain_loss_pct = db.fetch(
            "SELECT (sum(market_value) - sum(cost))/sum(cost) " + "FROM holdings_df",
            return_df=False,
        )[0][0]
        gain_loss_pct_formatted = f"{gain_loss_pct*100:,.2f}%"
        column.metric("Account Gain/Loss (%)", gain_loss_pct_formatted)
        return None

    def color_negative_red(value):
        """Color elements in a DataFrame green if positive and red if negative.

        Parameters
        ----------
        value : float
            The value to color.

        Returns
        -------
        str
            The CSS color property.

        """
        if value < 0:
            color = "red"
        elif value > 0:
            color = "green"
        else:
            color = "black"

        return f"color: {color}"

    def holdings_df_styled(accounts, index, column):
        """Display the styled holdings DataFrame for the given account.

        Parameters
        ----------
        accounts : list
            The list of accounts.
        index : int
            The index of the selected account.
        column : streamlit.delta_generator.DeltaGenerator
            The Streamlit column to display the DataFrame in.

        """
        holdings_df, future_holdings_df = Portfolio.get_holdings_df_filtered(
            accounts, index
        )
        holdings_df_columns = {
            "account_name": "Account",
            "ticker": "Ticker",
            "shares": "Shares",
            "price": "Price",
            "market_value": "Market Value",
            "cost": "Cost",
            "gain_loss": "Gain/Loss",
            "gain_loss_pct": "Gain/Loss %",
            "target_weight": "Target Weight",
            "current_weight": "Current Weight",
            "target_diff": "Target Diff",
        }
        column.markdown(
            '<div style="text-align: center;">'
            + "<strong>Before  Rebalance </strong>"
            + "</div>",
            unsafe_allow_html=True,
        )
        column.dataframe(
            holdings_df.loc[:, list(holdings_df_columns.keys())]
            .rename(columns=holdings_df_columns)
            .style.map(
                Portfolio.color_negative_red,
                subset=["Target Diff", "Gain/Loss", "Gain/Loss %"],
            )
            .format(
                {
                    "Target Weight": lambda x: f"{x:,.2f}%",
                    "Current Weight": lambda x: f"{x:,.2f}%",
                    "Target Diff": lambda x: f"{x:,.2f}%",
                    "Gain/Loss %": lambda x: f"{x:,.2f}%",
                    "Cost": lambda x: f"${x:,.2f}",
                    "Market Value": lambda x: f"${x:,.2f}",
                    "Price": lambda x: f"${x:,.2f}",
                    "Gain/Loss": lambda x: f"${x:,.2f}",
                    "Shares": lambda x: f"{x:,.4f}",
                }
            )
        )
        return None

    def future_holdings_df_styled(accounts, index, column):
        """Display the styled future holdings DataFrame for the given account.

        Parameters
        ----------
        accounts : list
            The list of accounts.
        index : int
            The index of the selected account.
        column : streamlit.delta_generator.DeltaGenerator
            The Streamlit column to display the DataFrame in.

        """
        holdings_df, future_holdings_df = Portfolio.get_holdings_df_filtered(
            accounts, index
        )
        holdings_df_columns = {
            "account_name": "Account",
            "ticker": "Ticker",
            "shares": "Shares",
            "price": "Price",
            "market_value": "Market Value",
            "cost": "Cost",
            "gain_loss": "Gain/Loss",
            "gain_loss_pct": "Gain/Loss %",
            "target_weight": "Target Weight",
            "current_weight": "Current Weight",
            "target_diff": "Target Diff",
        }
        column.markdown(
            '<div style="text-align: center;">'
            + "<strong>After Rebalance </strong>"
            + "</div>",
            unsafe_allow_html=True,
        )
        column.dataframe(
            future_holdings_df.loc[:, list(holdings_df_columns.keys())]
            .rename(columns=holdings_df_columns)
            .style.map(
                Portfolio.color_negative_red,
                subset=["Target Diff", "Gain/Loss", "Gain/Loss %"],
            )
            .format(
                {
                    "Target Weight": lambda x: f"{x:,.2f}%",
                    "Current Weight": lambda x: f"{x:,.2f}%",
                    "Target Diff": lambda x: f"{x:,.2f}%",
                    "Gain/Loss %": lambda x: f"{x:,.2f}%",
                    "Cost": lambda x: f"${x:,.2f}",
                    "Market Value": lambda x: f"${x:,.2f}",
                    "Price": lambda x: f"${x:,.2f}",
                    "Gain/Loss": lambda x: f"${x:,.2f}",
                    "Shares": lambda x: f"{x:,.4f}",
                }
            )
        )
        return None

    def grouped_bar_chart(accounts, index, column):
        """Display a grouped bar chart comparing target weight differences before and after rebalance.

        Parameters
        ----------
        accounts : list
            The list of accounts.
        index : int
            The index of the selected account.
        column : streamlit.delta_generator.DeltaGenerator
            The Streamlit column to display the chart in.

        """
        holdings_df, future_holdings_df = Portfolio.get_holdings_df_filtered(
            accounts, index
        )
        combined_df = db.fetch(get_query_string("combined_holdings"))

        with open("app/grouped_bar_chart.json") as file:
            vega_chart = json.load(file)

        column.markdown(
            '<div style="text-align: left;">'
            + "<strong>Comparison of Target Weight Differences Before vs "
            + "After Rebalance</strong>"
            + "</div>",
            unsafe_allow_html=True,
        )

        column.vega_lite_chart(combined_df, vega_chart, use_container_width=True)

        return None

    def orders_df_styled(accounts, index, column):
        """Display the styled orders DataFrame for the given account.

        Parameters
        ----------
        accounts : list
            The list of accounts.
        index : int
            The index of the selected account.
        column : streamlit.delta_generator.DeltaGenerator
            The Streamlit column to display the DataFrame in.

        """
        holdings_df, future_holdings_df = Portfolio.get_holdings_df_filtered(
            accounts, index
        )
        combined_df = db.fetch(get_query_string("combined_holdings"))  # noqa: F841

        orders_df = db.fetch(get_query_string("select_orders"))

        column.markdown(
            '<div style="text-align: left;">'
            + "<strong>Orders Needed For Rebalance</strong>"
            + "</div>",
            unsafe_allow_html=True,
        )

        column.dataframe(
            orders_df.style.map(
                Portfolio.color_negative_red, subset=["Trade Amount"]
            ).format(
                {
                    "Price": lambda x: f"${x:,.2f}",
                    "Trade Amount": lambda x: f"${x:,.2f}",
                    "Shares": lambda x: f"{x:,.4f}",
                }
            )
        )
        return None
