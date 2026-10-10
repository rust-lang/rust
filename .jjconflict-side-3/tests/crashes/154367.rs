//@ known-bug: #154367
//@ compile-flags: -Copt-level=0
#![feature(unsafe_binders)]

fn main() { panic::<unsafe<'a> &'a ()>(); }
fn panic<T>() { panic!() }
