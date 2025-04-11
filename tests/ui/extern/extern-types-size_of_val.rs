//@ check-fail
#![feature(extern_types)]

use std::mem::{align_of_val, size_of_val};

extern "C" {
    type A;
}

fn main() {
    let x: &A = unsafe { &*(1usize as *const A) };

    size_of_val(x);
    //~^ ERROR the size for values of type `A` cannot be known
    align_of_val(x);
    //~^ ERROR the size for values of type `A` cannot be known
}
