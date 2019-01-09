// compile-flags: -C no-prepopulate-passes

// This file tests that we don't generate any code for saturation if
// -Z saturating-float-casts is not enabled.

#![crate_type = "lib"]

// CHECK-LABEL: @f32_to_u32
#[no_mangle]
pub fn f32_to_u32(x: f32) -> u32 {
    // CHECK: fptoui
    // CHECK-NOT: fcmp
    // CHECK-NOT: icmp
    // CHECK-NOT: select
    x as u32
}

// CHECK-LABEL: @f32_to_i32
#[no_mangle]
pub fn f32_to_i32(x: f32) -> i32 {
    // CHECK: fptosi
    // CHECK-NOT: fcmp
    // CHECK-NOT: icmp
    // CHECK-NOT: select
    x as i32
}

#[no_mangle]
pub fn f64_to_u16(x: f64) -> u16 {
    // CHECK: fptoui
    // CHECK-NOT: fcmp
    // CHECK-NOT: icmp
    // CHECK-NOT: select
    x as u16
}
