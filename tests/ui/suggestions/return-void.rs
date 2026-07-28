#![crate_type = "lib"]

pub fn foo() -> void {} //~ ERROR cannot find type `void`

pub fn bar(v: void) {} //~ ERROR cannot find type `void`
