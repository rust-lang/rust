#![feature(register_tool)]

#![register_tool(1)]
//~^ ERROR malformed `register_tool` attribute input

fn main() {}
