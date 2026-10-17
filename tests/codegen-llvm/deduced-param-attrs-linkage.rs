// Test that we do not deduce parameter attributes when the function has a linkage attribute that
// indicates a different definition may be swapped in at link-time.

//@ compile-flags: -O -Cno-prepopulate-passes
#![crate_type = "lib"]
#![feature(linkage)]

pub struct Big {
    pub blah: [i32; 1024],
}

// CHECK-LABEL: @external(ptr {{.*}}readonly {{.*}}captures(none) {{.*}}%_b)
#[unsafe(no_mangle)]
pub fn external(_b: Big) {}

// CHECK-LABEL: define linkonce void @linkonce(
// CHECK-NOT:   readonly
// CHECK-NOT:   captures(none)
// CHECK-SAME:  %_b)
#[unsafe(no_mangle)]
#[linkage = "linkonce"]
pub fn linkonce(_b: Big) {}

// CHECK-LABEL: define linkonce_odr void @linkonce_odr(
// CHECK-NOT:   readonly
// CHECK-NOT:   captures(none)
// CHECK-SAME:  %_b)
#[unsafe(no_mangle)]
#[linkage = "linkonce_odr"]
pub fn linkonce_odr(_b: Big) {}

// CHECK-LABEL: define weak void @weak(
// CHECK-NOT:   readonly
// CHECK-NOT:   captures(none)
// CHECK-SAME:  %_b)
#[unsafe(no_mangle)]
#[linkage = "weak"]
pub fn weak(_b: Big) {}

// CHECK-LABEL: define weak_odr void @weak_odr(
// CHECK-NOT:   readonly
// CHECK-NOT:   captures(none)
// CHECK-SAME:  %_b)
#[unsafe(no_mangle)]
#[linkage = "weak_odr"]
pub fn weak_odr(_b: Big) {}

// CHECK-LABEL: define available_externally void @available_externally(
// CHECK-NOT:   readonly
// CHECK-NOT:   captures(none)
// CHECK-SAME:  %_b)
#[unsafe(no_mangle)]
#[linkage = "available_externally"]
pub fn available_externally(_b: Big) {}
