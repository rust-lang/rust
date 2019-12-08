#![feature(register_attr)]
#![feature(register_tool)]

#![register_attr] //~ ERROR malformed `register_attr` attribute input
#![register_tool] //~ ERROR malformed `register_tool` attribute input

#![register_attr(a::b)] //~ ERROR `register_attr` only accepts identifiers
#![register_tool(a::b)] //~ ERROR `register_tool` only accepts identifiers

#![register_attr(attr, attr)] //~ ERROR attribute `attr` was already registered
#![register_tool(tool, tool)] //~ ERROR tool `tool` was already registered

fn main() {}
