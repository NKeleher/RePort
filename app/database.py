import duckdb


def get_query_string(file_path: str) -> str:
    """Read the SQL query from the specified file.

    Parameters
    ----------
    file_path : str
        The path to the SQL file.

    Returns
    -------
    str
        The SQL query as a string.

    """
    with open("app/data/queries/" + file_path + ".sql") as file:
        return file.read()


class DuckDB:
    """Class to handle DuckDB database operations."""

    def __init__(self):
        """Initialize the DuckDB connection."""
        self.database = ":memory:"
        self.connection = duckdb.connect(self.database)

    def query(self, sql_statement: str, parameters=None):
        """Execute a SQL query.

        Parameters
        ----------
        sql_statement : str
            The SQL query to execute.
        parameters : list, optional
            The parameters for the SQL query, by default None.

        """
        if parameters is None:
            self.connection.execute(sql_statement)
        else:
            self.connection.executemany(sql_statement, parameters)
        self.connection.commit()

    def fetch(self, sql_statement: str, return_df=True):
        """Fetch the results of a SQL query.

        Parameters
        ----------
        sql_statement : str
            The SQL query to execute.
        return_df : bool, optional
            Whether to return the results as a DataFrame, by default True.

        Returns
        -------
        pd.DataFrame or list
            The query results as a DataFrame if return_df is True, otherwise as a list.

        """
        if return_df:
            return self.connection.execute(sql_statement).df()
        else:
            return self.connection.execute(sql_statement).fetchall()

    def crud(self, operation, table, params):
        """Perform CRUD operations on the specified table.

        Parameters
        ----------
        operation : str
            The CRUD operation to perform ('Add', 'Delete', 'Update').
        table : str
            The table to perform the operation on ('cash', 'holdings').
        params : list
            The parameters for the SQL query.

        """
        # TO DO: Should be better way to abstract this to I don't repeat myself
        # TO DO: Find better way to create initial tables
        if table == "cash":
            if operation == "Add":
                self.query(get_query_string("insert_cash_values"), params)
            if operation == "Delete":
                self.query(get_query_string("delete_cash"), params)
            if operation == "Update":
                self.query(get_query_string("update_cash"), params)
        if table == "holdings":
            # TO DO: Add error handling for invalid values (e.g. 0 shares)
            if operation == "Add":
                self.query(get_query_string("insert_holdings_values"), params)
            if operation == "Delete":
                self.query(get_query_string("delete_holding"), params)
            if operation == "Update":
                self.query(get_query_string("update_holding"), params)
        return
