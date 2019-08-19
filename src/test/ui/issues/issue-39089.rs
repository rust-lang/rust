// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
fn f<T: ?for<'a> Sized>() {}

fn main() {}
