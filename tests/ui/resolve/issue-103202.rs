#![allow(self_lifetime_elision_not_applicable)]
struct S {}

impl S {
    fn f(self: &S::x) {} //~ ERROR ambiguous associated type
}

fn main() {}
