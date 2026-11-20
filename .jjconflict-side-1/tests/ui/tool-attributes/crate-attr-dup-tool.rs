//@ check-pass
//@ compile-flags: -Z crate-attr=feature(register_tool) -Z crate-attr=register_tool(foo)
//@ compile-flags: -Z crate-attr=register_attribute_tool(bar) -Z crate-attr=register_lint_tool(baz)
//@ compile-flags: -A duplicate_features -A duplicate_tools
#![feature(register_tool)]
#![register_tool(foo)]
#![register_attribute_tool(bar)]
#![register_lint_tool(baz)]

#[foo::foo]
#[bar::bar]
#[allow(foo::baz, baz::baz)]
fn main() {}
