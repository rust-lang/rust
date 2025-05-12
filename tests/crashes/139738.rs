//@ known-bug: #139738
#![feature(generic_const_exprs)]
fn b<'a>() -> impl IntoIterator<[(); (|_: &'a u8| 0, 0).1]> {}
