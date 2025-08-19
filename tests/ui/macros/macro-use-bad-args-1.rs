#![no_std]

#[macro_use(foo(bar))]  //~ ERROR malformed `macro_use` attribute input
extern crate std;

fn main() {}
