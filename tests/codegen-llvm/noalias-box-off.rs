//@ compile-flags: -Copt-level=3 -Z box-noalias=no

#![crate_type = "lib"]

// CHECK-LABEL: @box_should_not_have_noalias_if_disabled(
// CHECK-NOT: noalias
// CHECK-SAME: %foo)
#[no_mangle]
pub fn box_should_not_have_noalias_if_disabled(foo: Box<u8>) {
    drop(foo);
}
