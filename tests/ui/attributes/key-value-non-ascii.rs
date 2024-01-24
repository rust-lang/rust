#![feature(rustc_attrs)]

#[rustc_dummy = b'ﬃ'] //~ ERROR non-ASCII character in byte literal
fn main() {}
