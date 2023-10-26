#![feature(rustc_attrs)]

#![rustc_dummy=5z] //~ ERROR unexpected expression: `5z`
fn main() {}
