#![feature(register_tool)]
#![register_tool(foo::bar)] //~ ERROR only accepts identifiers

fn main() {}
