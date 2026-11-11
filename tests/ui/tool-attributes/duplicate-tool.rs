//@ check-pass
#![feature(register_tool)]
#![warn(duplicate_tools)]
// Register a tool multiple times is okay.
#![register_tool(foo)]
#![register_tool(foo)] //~ WARN [duplicate_tools]
#![register_tool(bar)]
#![register_attribute_tool(bar)] //~ WARN [duplicate_tools]
#![register_tool(baz)]
#![register_lint_tool(baz)] //~ WARN [duplicate_tools]
#![register_attribute_tool(qux)]
#![register_attribute_tool(qux)] //~ WARN [duplicate_tools]
#![register_lint_tool(quux)]
#![register_lint_tool(quux)] //~ WARN [duplicate_tools]

fn main() {}
