// check-pass
// compile-flags: --cfg foo

#![feature(register_attr)]
#![feature(register_tool)]

#![register_attr(attr)]
#![register_tool(tool)]
#![register_tool(rustfmt, clippy)] // OK
#![cfg_attr(foo, register_attr(conditional_attr))]
#![cfg_attr(foo, register_tool(conditional_tool))]

#[attr]
#[tool::attr]
#[rustfmt::attr]
#[clippy::attr]
#[conditional_attr]
#[conditional_tool::attr]
fn main() {}
