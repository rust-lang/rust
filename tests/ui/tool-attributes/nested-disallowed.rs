#![feature(register_tool)]
#![register_tool(foo::bar)] //~ ERROR tools are always a single identifier, not paths with multiple segments

fn main() {}
