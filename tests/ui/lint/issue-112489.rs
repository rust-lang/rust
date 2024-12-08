//@ check-pass
use std::borrow::Borrow;

struct S;

trait T: Sized {
    fn foo(self) {}
}

impl T for S {}
impl T for &S {}

fn main() {
    let s = S;
    s.borrow().foo();
    s.foo();
}
