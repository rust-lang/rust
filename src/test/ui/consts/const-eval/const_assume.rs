// check-pass

// Check that `const_assume` feature allow `assume` intrinsic
// to be used in const contexts.

#![feature(core_intrinsics, const_assume)]

extern crate core;

use core::intrinsics::assume;

pub const unsafe fn foo(x: usize, y: usize) -> usize {
    assume(y != 0);
    x / y
}

fn main() {}
