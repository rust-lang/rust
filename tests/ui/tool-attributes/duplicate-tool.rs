//@ check-pass
#![feature(register_tool)]
// Register a tool multiple times is okay.
#![register_tool(foo)]
#![register_tool(foo)]
#![register_tool(bar)]
#![register_attribute_tool(bar)]
#![register_tool(baz)]
#![register_lint_tool(baz)]
#![register_attribute_tool(qux)]
#![register_attribute_tool(qux)]
#![register_lint_tool(quux)]
#![register_lint_tool(quux)]

fn main() {}
