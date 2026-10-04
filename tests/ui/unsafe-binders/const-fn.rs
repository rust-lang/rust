//@ compile-flags: -Znext-solver
//@ check-pass

// Wrapping and unwrapping work in const fns and const eval.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::unsafe_binder::{unwrap_binder, wrap_binder};

const fn round_trip(x: &u8) -> u8 {
    let b: unsafe<'a> &'a u8 = unsafe { wrap_binder!(x) };
    unsafe { *unwrap_binder!(b) }
}

const C: u8 = round_trip(&3);
const _: () = assert!(C == 3);

fn main() {}
