//@ check-pass
#![feature(register_tool)]
#![register_tool(foo)]
#![foo::is_a_tool]

fn main() {}
