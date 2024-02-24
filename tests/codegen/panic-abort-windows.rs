// This test is for *-windows only.
//@ only-windows

//@ compile-flags: -C no-prepopulate-passes -C panic=abort -O

#![crate_type = "lib"]

// CHECK: Function Attrs: nounwind uwtable
// CHECK-NEXT: define void @normal_uwtable()
#[no_mangle]
pub fn normal_uwtable() {
}

// CHECK: Function Attrs: nounwind uwtable
// CHECK-NEXT: define void @extern_uwtable()
#[no_mangle]
pub extern fn extern_uwtable() {
}
