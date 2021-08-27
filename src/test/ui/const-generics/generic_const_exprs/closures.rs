#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
fn test<const N: usize>() -> [u8; N + (|| 42)()] {}
//~^ ERROR overly complex generic constant

fn main() {}
