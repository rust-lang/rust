//! Regression test for <https://github.com/rust-lang/rust/issues/161251>.
//@ compile-flags: -Znext-solver=globally
//@ check-pass

#![feature(pointer_is_aligned_to, transmutability)]
use std::mem::{Assume, TransmuteFrom};

fn main() {
    let src: &[u8; 2] = &[0xFF, 0xFF];

    let maybe_dst: Option<&u16> = if <*const _>::is_aligned_to(src, align_of::<u16>()) {
        Some(unsafe {
            <_ as TransmuteFrom<_, { Assume::ALIGNMENT }>>::transmute(src)
        })
    } else {
        None
    };
}
