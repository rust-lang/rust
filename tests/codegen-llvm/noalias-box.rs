//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @box_should_have_noalias_by_default(
// CHECK: noalias
#[no_mangle]
pub fn box_should_have_noalias_by_default(_b: Box<u8>) {}
