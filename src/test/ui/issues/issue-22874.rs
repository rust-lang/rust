struct Table {
    rows: [[String]],
    //~^ ERROR the size for values of type
}

fn f(table: &Table) -> &[String] {
    &table.rows[0]
}

fn main() {}
