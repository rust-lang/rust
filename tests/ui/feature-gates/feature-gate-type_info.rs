#![feature(register_tool)]
#![register_tool(aaa(asdf))]
//~^ ERROR: tool attribute options are currently unstable
#![register_tool(bbb(reflect))]
//~^ ERROR: tool attribute options are currently unstable

fn main() {}
