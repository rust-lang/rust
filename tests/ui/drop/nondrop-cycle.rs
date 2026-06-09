//@ run-pass

use std::cell::Cell;

struct C<'a> {
    p: Cell<Option<&'a C<'a>>>,
}

impl<'a> C<'a> {
    fn new() -> C<'a> { C { p: Cell::new(None) } }
}

fn f1() {
    let (c1, c2) = (C::new(), C::new());
    c1.p.set(Some(&c2));
    c2.p.set(Some(&c1));
}

fn f2() {
    let (c1, c2);
    c1 = C::new();
    c2 = C::new();
    c1.p.set(Some(&c2));
    c2.p.set(Some(&c1));
}

fn main() {
    f1();
    f2();
}
