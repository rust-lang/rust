// This is a regression test for <https://github.com/rust-lang/rust/issues/139462>.
//@ check-pass
//@ compile-flags: -Cdebuginfo=2
//@ ignore-backends: gcc
#![feature(unsafe_binders)]
//~^ WARN the feature `unsafe_binders` is incomplete

use std::unsafe_binder::wrap_binder;
fn main() {
    let foo = 0;
    let foo: unsafe<'a> &'a u32 = unsafe { wrap_binder!(&foo) };
}
