#![feature(rustc_attrs)]

#[rustc_dummy = b"ﬃ.rs"] //~ ERROR non-ASCII character in byte string literal
fn main() {}
