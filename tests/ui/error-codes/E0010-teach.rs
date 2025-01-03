//@ compile-flags: -Z teach

#![allow(warnings)]

const CON: Vec<i32> = vec![1, 2, 3]; //~ ERROR cannot call non-const
//~| ERROR cannot call non-const
fn main() {}
