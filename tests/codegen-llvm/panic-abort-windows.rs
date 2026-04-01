// This test is for *-windows only.
//@ only-windows

//@ compile-flags: -C no-prepopulate-passes -C panic=abort -Copt-level=3

#![crate_type = "lib"]

// CHECK: Function Attrs: nounwind uwtable
// CHECK-NEXT: define void @normal_uwtable()
#[no_mangle]
pub fn normal_uwtable() {}

// CHECK: Function Attrs: nounwind uwtable
// CHECK-NEXT: define void @extern_uwtable()
#[no_mangle]
pub extern "C" fn extern_uwtable() {}
