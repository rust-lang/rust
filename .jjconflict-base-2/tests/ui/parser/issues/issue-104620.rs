#![feature(rustc_attrs)]

#![rustc_dummy=5z] //~ ERROR invalid suffix `z` for number literal
fn main() {}
