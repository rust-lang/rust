#![feature(register_tool)]
// Register predefined tool or "rustc" is error.
#![register_tool(clippy)] //~ ERROR predefined
                          //~^ ERROR predefined
#![register_attribute_tool(miri)] //~ ERROR predefined
#![register_lint_tool(rustfmt)] //~ ERROR predefined
#![register_tool(diagnostic)] //~ ERROR predefined
                              //~^ ERROR predefined
#![register_attribute_tool(rust_analyzer)] //~ ERROR predefined
#![register_tool(rustc)] //~ ERROR reserved
                         //~^ ERROR reserved

fn main() {}
