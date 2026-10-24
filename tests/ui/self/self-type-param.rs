//@ check-pass
#![allow(dead_code)]

trait MyTrait {
    fn f(&self) -> Self;
}

struct S {
    x: isize
}

impl MyTrait for S {
    fn f(&self) -> S {
        S { x: 3 }
    }
}

pub fn main() {}
