#![feature(register_tool)]

#![register_tool(1)]
//~^ ERROR malformed `register_tool` attribute input
#![register_tool(_)]
//~^ ERROR expected identifier, found reserved identifier `_`
//~| ERROR expected identifier, found reserved identifier `_`
//~| ERROR expected identifier, found reserved identifier `_`
//~| ERROR malformed `register_tool` attribute input

// Special path keywords cannot be used.
#![register_tool(crate)]
//~^ ERROR malformed `register_tool` attribute input
#![register_tool(self)]
//~^ ERROR malformed `register_tool` attribute input
#![register_tool(Self)]
//~^ ERROR malformed `register_tool` attribute input
#![register_tool(super)]
//~^ ERROR malformed `register_tool` attribute input

// These are okay
#![register_tool(r#type)]
#![register_tool(铁锈)]

fn main() {}
