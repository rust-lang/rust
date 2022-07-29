// run-rustfix
#![allow(unused_imports)]

fn main() {}

Use std::ptr::read;  //~ ERROR keyword `use` is written in a wrong case
USE std::ptr::write; //~ ERROR keyword `use` is written in a wrong case
