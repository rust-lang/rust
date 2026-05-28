//@ check-pass
//@ compile-flags: -Z crate-attr=feature(register_tool) -Z crate-attr=register_tool(foo)

#[allow(foo::bar)]
fn main() {}
