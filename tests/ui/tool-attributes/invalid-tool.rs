#![feature(register_tool)]

#![register_tool(1)]
//~^ ERROR `register_tool` only accepts identifiers

fn main() {}
