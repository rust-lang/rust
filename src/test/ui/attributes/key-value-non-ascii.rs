#![feature(rustc_attrs)]

#[rustc_dummy = b"ﬃ.rs"] //~ ERROR byte constant must be ASCII
fn main() {}
