// Verify that algebraic intrinsics generate the correct LLVM calls for f32

#![crate_type = "lib"]
#![feature(float_algebraic)]

// CHECK-LABEL: float @f32_algebraic_add(
#[no_mangle]
pub fn f32_algebraic_add(a: f32, b: f32) -> f32 {
    // CHECK: fadd reassoc nsz arcp contract float {{(%a, %b)|(%b, %a)}}
    a.algebraic_add(b)
}

// CHECK-LABEL: float @f32_algebraic_sub(
#[no_mangle]
pub fn f32_algebraic_sub(a: f32, b: f32) -> f32 {
    // CHECK: fsub reassoc nsz arcp contract float %a, %b
    a.algebraic_sub(b)
}

// CHECK-LABEL: float @f32_algebraic_mul(
#[no_mangle]
pub fn f32_algebraic_mul(a: f32, b: f32) -> f32 {
    // CHECK: fmul reassoc nsz arcp contract float {{(%a, %b)|(%b, %a)}}
    a.algebraic_mul(b)
}

// CHECK-LABEL: float @f32_algebraic_div(
#[no_mangle]
pub fn f32_algebraic_div(a: f32, b: f32) -> f32 {
    // CHECK: fdiv reassoc nsz arcp contract float %a, %b
    a.algebraic_div(b)
}

// CHECK-LABEL: float @f32_algebraic_rem(
#[no_mangle]
pub fn f32_algebraic_rem(a: f32, b: f32) -> f32 {
    // CHECK: frem reassoc nsz arcp contract float %a, %b
    a.algebraic_rem(b)
}
