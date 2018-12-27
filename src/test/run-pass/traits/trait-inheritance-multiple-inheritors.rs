// run-pass
#![allow(dead_code)]

trait A { fn a(&self) -> isize; }
trait B: A { fn b(&self) -> isize; }
trait C: A { fn c(&self) -> isize; }

struct S { bogus: () }

impl A for S { fn a(&self) -> isize { 10 } }
impl B for S { fn b(&self) -> isize { 20 } }
impl C for S { fn c(&self) -> isize { 30 } }

// Both B and C inherit from A
fn f<T:B + C>(x: &T) {
    assert_eq!(x.a(), 10);
    assert_eq!(x.b(), 20);
    assert_eq!(x.c(), 30);
}

pub fn main() {
    f(&S { bogus: () })
}
