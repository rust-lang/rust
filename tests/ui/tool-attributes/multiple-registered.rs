//@ check-pass

#![feature(register_tool)]
#![register_tool(foo, bar, baz)]

#[allow(foo::a, bar::b, baz::c)]
fn main() {}
