//@ check-pass

trait Database: for<'r> HasValueRef<'r, Database = Self> {}

trait HasValueRef<'r> {
    type Database: Database;
}

struct Any;

impl Database for Any {}

impl<'r> HasValueRef<'r> for Any {
    // Make sure we don't have issues when the GAT assumption
    // `<Any as HasValue<'r>>::Database = Any` isn't universally
    // parameterized over `'r`.
    type Database = Any;
}

fn main() {}
