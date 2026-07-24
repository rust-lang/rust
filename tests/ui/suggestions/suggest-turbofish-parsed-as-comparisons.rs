// Issue #81816.
use std::sync::Arc;
use std::sync::RwLock;
use std::collections::HashMap;

struct S;
impl S {
    fn foo(&self, _: HashMap<i32, i64>) {}
}
fn main() {
    let _ = Arc::new(RwLock::new(HashMap<i32, i64>::default()));
    //~^ ERROR can't compare two types
    let _ = S.foo(HashMap<i32, i64>::default());
    //~^ ERROR can't compare two types
}
