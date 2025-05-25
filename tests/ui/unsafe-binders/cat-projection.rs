//@ check-pass

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::unsafe_binder::unwrap_binder;

#[derive(Copy, Clone)]
pub struct S([usize; 8]);

// Regression test for <https://github.com/rust-lang/rust/issues/141418>.
pub fn by_value(x: unsafe<'a> S) -> usize {
    unsafe { (|| unwrap_binder!(x).0[0])() }
}

// Regression test for <https://github.com/rust-lang/rust/issues/141417>.
pub fn by_ref(x: unsafe<'a> &'a S) -> usize {
    unsafe { (|| unwrap_binder!(x).0[0])() }
}

fn main() {}
