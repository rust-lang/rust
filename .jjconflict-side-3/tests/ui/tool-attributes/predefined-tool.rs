#![feature(register_tool)]
// Registering predefined tool is okay.
#![register_tool(clippy)]
#![register_attribute_tool(miri)]
#![register_lint_tool(rustfmt)]
#![register_tool(diagnostic)]
#![register_attribute_tool(rust_analyzer)]
// Registering "rustc" is an error.
#![register_tool(rustc)] //~ ERROR reserved

fn main() {}
