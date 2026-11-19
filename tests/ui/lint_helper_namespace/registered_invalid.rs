#![crate_type = "lib"]
#![feature(register_tool)]
#![register_tool(lint)]

#[lint::bogus] //~ ERROR the `lint::bogus` attribute is not recognized
struct Foo;
