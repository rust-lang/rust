//@ edition:2021
//@ check-pass

use core::ops::Deref;

struct A;
struct B;
struct C;

impl Deref for C {
    type Target = B;
    fn deref(&self) -> &Self::Target {
        &B
    }
}

impl Deref for B {
    type Target = A;
    fn deref(&self) -> &Self::Target {
        &A
    }
}

fn f(v: u8) {
    let _ = match v {
        0 => &C,
        1 => &B,
        _ => &A,
    };
}

fn main() {}
