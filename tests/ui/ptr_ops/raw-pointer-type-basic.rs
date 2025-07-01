//! Checks the basic usage of raw pointers (`*const isize`) as function argument and return types.

//@ run-pass

#![allow(dead_code)]

fn f(a: *const isize) -> *const isize {
    return a;
}

fn g(a: *const isize) -> *const isize {
    let b = f(a);
    return b;
}

pub fn main() {
    return;
}
