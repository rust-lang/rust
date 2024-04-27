//@ compile-flags: -Z mir-opt-level=3
//@ run-pass

use std::cell::Cell;

#[derive(Debug)]
struct B<'a> {
    a: [Cell<Option<&'a B<'a>>>; 2]
}

impl<'a> B<'a> {
    fn new() -> B<'a> {
        B { a: [Cell::new(None), Cell::new(None)] }
    }
}

fn f() {
    let b2 = B::new();
    b2.a[0].set(Some(&b2));
}

fn main() {
    f();
}
