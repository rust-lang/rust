// run-pass

#![deny(where_clauses_object_safety)]

trait Y {
    fn foo(&self) where Self: Send {}
    fn bar(&self) where Self: Send + Sync {}
}

impl Y for () {}

fn main() {
    <dyn Y + Send as Y>::foo(&());
    <dyn Y + Send + Sync as Y>::bar(&());
}
