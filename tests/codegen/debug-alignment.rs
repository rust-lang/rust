// Verifies that DWARF alignment is specified properly.
//
//@ compile-flags: -C debuginfo=2
#![crate_type = "lib"]

// CHECK: !DIGlobalVariable
// CHECK: align: 32
pub static A: u32 = 1;
