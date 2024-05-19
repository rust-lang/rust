//@ check-pass
#![allow(dead_code)]
fn f<T: ?for<'a> Sized>() {}

fn main() {}
