// compile-flags: -A known_tool::foo
// check-pass

#![cfg_attr(bootstrap, feature(register_tool))]
#![register_tool(known_tool)]

fn main() {}
