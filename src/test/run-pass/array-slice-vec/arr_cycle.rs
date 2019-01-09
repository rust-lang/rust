// run-pass

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
    let (b1, b2, b3);
    b1 = B::new();
    b2 = B::new();
    b3 = B::new();
    b1.a[0].set(Some(&b2));
    b1.a[1].set(Some(&b3));
    b2.a[0].set(Some(&b2));
    b2.a[1].set(Some(&b3));
    b3.a[0].set(Some(&b1));
    b3.a[1].set(Some(&b2));
}

fn main() {
    f();
}
