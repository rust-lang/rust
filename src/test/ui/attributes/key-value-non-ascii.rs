#![feature(rustc_attrs)]

#[rustc_dummy = b"ﬃ.rs"] //~ ERROR non-ASCII character in byte constant
fn main() {}
