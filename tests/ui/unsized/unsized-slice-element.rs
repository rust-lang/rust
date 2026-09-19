//! Regression test for <https://github.com/rust-lang/rust/issues/22874>.

struct Table {
    rows: [[String]],
    //~^ ERROR the size for values of type
}

fn f(table: &Table) -> &[String] {
    &table.rows[0]
    //~^ ERROR the size for values of type
}

fn main() {}
