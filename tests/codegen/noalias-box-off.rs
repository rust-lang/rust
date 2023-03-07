// compile-flags: -O -Z box-noalias=no

#![crate_type = "lib"]

// CHECK-LABEL: @box_should_not_have_noalias_if_disabled(
// CHECK-NOT: noalias
#[no_mangle]
pub fn box_should_not_have_noalias_if_disabled(_b: Box<u8>) {}
