#![no_std]

#[macro_use(foo="bar")]  //~ ERROR bad macro import
extern crate std;

fn main() {}
