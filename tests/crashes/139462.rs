//@ known-bug: #139462
//@ compile-flags: -Cdebuginfo=2
#![feature(unsafe_binders)]
use std::unsafe_binder::wrap_binder;
fn main() {
    let foo = 0;
    let foo: unsafe<'a> &'a u32 = unsafe { wrap_binder!(&foo) };
}
