#![allow(warnings)]

const CON: Vec<i32> = vec![1, 2, 3]; //~ ERROR E0010
//~| ERROR cannot call non-const fn
fn main() {}
