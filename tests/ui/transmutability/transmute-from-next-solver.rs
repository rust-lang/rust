//! Regression test for <https://github.com/rust-lang/rust/issues/161251>.
//@ compile-flags: -Znext-solver=globally
//@ check-pass

#![feature(transmutability)]
use std::mem::TransmuteFrom;

fn main() {
    let bytes: &[u8; 4] = unsafe { TransmuteFrom::<&i32>::transmute(&1i32) };
}
