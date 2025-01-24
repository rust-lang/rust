//@ run-pass

#![feature(arbitrary_self_types)]

pub struct A;

impl A {
    pub fn f(self: B) -> i32 { 1 }
}

pub struct B(A);

impl core::ops::Receiver for B {
    type Target = A;
}

struct C;

struct D;

impl C {
    fn weird(self: D) -> i32 { 3 }
}

impl core::ops::Receiver for D {
    type Target = C;
}

fn main() {
    assert_eq!(B(A).f(), 1);
    assert_eq!(D.weird(), 3);
}
