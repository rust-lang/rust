// Verifies that "EnableSplitLTOUnit" module flag is added.
//
//@ compile-flags: -Clto -Ctarget-feature=-crt-static -Zsplit-lto-unit

#![crate_type = "lib"]

pub fn foo() {}

// CHECK: !{{[0-9]+}} = !{i32 4, !"EnableSplitLTOUnit", i32 1}
