#![feature(register_tool)]
#![register_tool(foo::bar)] //~ ERROR malformed `register_tool` attribute input

fn main() {}
