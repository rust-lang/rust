#![no_std]

#[allow(unused_extern_crates)]
#[macro_use(foo="bar")]  //~ ERROR bad macro import
extern crate std;

fn main() {}
