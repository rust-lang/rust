//@ compile-flags: -Z teach

#![allow(warnings)]

const CON: Vec<i32> = vec![1, 2, 3]; //~ ERROR E0010
//~| ERROR is not yet stable as a const fn
fn main() {}
