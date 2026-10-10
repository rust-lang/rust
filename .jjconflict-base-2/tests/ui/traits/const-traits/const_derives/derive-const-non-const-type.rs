#![feature(const_default, derive_const)]

pub struct A;

impl Default for A {
    fn default() -> A {
        A
    }
}

#[derive_const(Default)]
pub struct S(A);
//~^ ERROR: cannot call non-const associated function

fn main() {}
