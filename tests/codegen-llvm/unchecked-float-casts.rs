// This file tests that we don't generate any code for saturation when using the
// unchecked intrinsics.

//@ compile-flags: -C opt-level=3
//@ ignore-wasm32 the wasm target is tested in `wasm_casts_*`

#![crate_type = "lib"]

// CHECK-LABEL: @f32_to_u32
#[no_mangle]
pub fn f32_to_u32(x: f32) -> u32 {
    // CHECK: fptoui
    // CHECK-NOT: fcmp
    // CHECK-NOT: icmp
    // CHECK-NOT: select
    unsafe { x.to_int_unchecked() }
}

// CHECK-LABEL: @f32_to_i32
#[no_mangle]
pub fn f32_to_i32(x: f32) -> i32 {
    // CHECK: fptosi
    // CHECK-NOT: fcmp
    // CHECK-NOT: icmp
    // CHECK-NOT: select
    unsafe { x.to_int_unchecked() }
}

#[no_mangle]
pub fn f64_to_u16(x: f64) -> u16 {
    // CHECK: fptoui
    // CHECK-NOT: fcmp
    // CHECK-NOT: icmp
    // CHECK-NOT: select
    unsafe { x.to_int_unchecked() }
}
