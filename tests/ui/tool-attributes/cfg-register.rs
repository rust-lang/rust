//@ check-pass
//
//@revisions: tool split_attr_lint

#![feature(register_tool)]
#![cfg_attr(tool, register_tool(foo, bar))]
#![cfg_attr(split_attr_lint, register_attribute_tool(foo))]
#![cfg_attr(split_attr_lint, register_lint_tool(bar))]

#[foo::a]
#[allow(bar::b)]
fn main() {}
