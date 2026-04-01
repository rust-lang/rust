// Check that the `'_` used in structs/enums gives an error.

use std::fmt::Debug;

fn foo(x: &u32, y: &u32) -> &'_ u32 { loop { } } //~ ERROR missing lifetime specifier

fn main() { }
