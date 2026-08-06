//@ build-pass
#![allow(internal_features)]
#![feature(rustc_attrs)]

#[rustc_force_inline]
fn f() {}
fn g<T: FnOnce()>(t: T) { t(); }

fn main() { g(f); }
