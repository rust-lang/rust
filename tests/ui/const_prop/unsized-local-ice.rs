//@ build-pass
//! Regression test for <https://github.com/rust-lang/rust/issues/68538>.
#![feature(unsized_fn_params)]

pub fn take_unsized_slice(s: [u8]) {
    s[0];
}

fn main() {}
