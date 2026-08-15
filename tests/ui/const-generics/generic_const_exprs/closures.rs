#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
fn test<const N: usize>() -> [u8; N + (|| 42)()] {}
//~^ ERROR cycle detected when building an abstract representation

fn main() {}
