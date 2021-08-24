#![deny(unused)]

#![feature(register_attr)]
#![feature(register_tool)]

#[register_attr(attr)] //~ ERROR crate-level attribute should be an inner attribute
#[register_tool(tool)] //~ ERROR crate-level attribute should be an inner attribute
fn main() {}
