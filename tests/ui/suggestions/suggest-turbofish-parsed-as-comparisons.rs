// Issue #81816.
use std::sync::Arc;
use std::sync::RwLock;
use std::collections::HashMap;

struct S;
impl S {
    fn foo(&self, _: HashMap<i32, i64>) {}
    fn bar(&self, _: Many<i32, i32, i32, i32>) {}
}
fn bar(_: Many<i32, i32, i32, i32>) {}
struct Many<A, B, C, D> {
    a: A,
    b: B,
    c: C,
    d: D,
}

impl<A, B, C, D> Many<A, B, C, D> {
    fn new() -> Self { todo!() }
}

fn main() {
    let _ = Arc::new(RwLock::new(HashMap<i32, i64>::default()));
    //~^ ERROR can't compare two types
    let _ = S.foo(HashMap<i32, i64>::default());
    //~^ ERROR can't compare two types
    let _ = bar(Many<i32, i32, i32, i32>::new());
    //~^ ERROR can't compare two types
    let _ = S.bar(Many<i32, i32, i32, i32>::new());
    //~^ ERROR can't compare two types
    let _ = bar(Many<i32, i32, i32, i32>::new(), Many<i32, i32, i32, i32>::new());
    //~^ ERROR can't compare two types
    //~| ERROR can't compare two types
    let _ = S.bar(Many<i32, i32, i32, i32>::new(), Many<i32, i32, i32, i32>::new());
    //~^ ERROR can't compare two types
    //~| ERROR can't compare two types
    let _ = bar(1, 2, Many<i32, i32, i32, i32>::new());
    //~^ ERROR can't compare two types
    let _ = S.bar(1, 2, Many<i32, i32, i32, i32>::new());
    //~^ ERROR can't compare two types
    let _ = bar(a, b, Many<i32, i32, i32, i32>::new(), c, d);
    //~^ ERROR can't compare two types
    //~| ERROR cannot find value `a` in this scope
    //~| ERROR cannot find value `b` in this scope
    //~| ERROR cannot find value `c` in this scope
    //~| ERROR cannot find value `d` in this scope
    let _ = S.bar(a, b, Many<i32, i32, i32, i32>::new(), c, d);
    //~^ ERROR can't compare two types
    //~| ERROR cannot find value `a` in this scope
    //~| ERROR cannot find value `b` in this scope
    //~| ERROR cannot find value `c` in this scope
    //~| ERROR cannot find value `d` in this scope
}
