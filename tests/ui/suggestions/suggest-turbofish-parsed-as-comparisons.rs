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
    //~^ ERROR expected value, found struct `HashMap` [E0423]
    //~| ERROR expected value, found builtin type `i32` [E0423]
    //~| ERROR expected value, found builtin type `i64` [E0423]
    //~| ERROR cannot find external crate `default` in the crate root [E0425]
    //~| ERROR this function takes 1 argument but 2 arguments were supplied [E0061]
    let _ = S.foo(HashMap<i32, i64>::default());
    //~^ ERROR expected value, found struct `HashMap` [E0423]
    //~| ERROR expected value, found builtin type `i32` [E0423]
    //~| ERROR expected value, found builtin type `i64` [E0423]
    //~| ERROR cannot find external crate `default` in the crate root [E0425]
    //~| ERROR this method takes 1 argument but 2 arguments were supplied [E0061]
}
