// Verify that algebraic intrinsics generate the correct LLVM calls for f16

#![crate_type = "lib"]
#![feature(f16)]
#![feature(float_algebraic)]

// CHECK-LABEL: half @f16_algebraic_add(
#[no_mangle]
pub fn f16_algebraic_add(a: f16, b: f16) -> f16 {
    // CHECK: fadd reassoc nsz arcp contract half {{(%a, %b)|(%b, %a)}}
    a.algebraic_add(b)
}

// CHECK-LABEL: half @f16_algebraic_sub(
#[no_mangle]
pub fn f16_algebraic_sub(a: f16, b: f16) -> f16 {
    // CHECK: fsub reassoc nsz arcp contract half %a, %b
    a.algebraic_sub(b)
}

// CHECK-LABEL: half @f16_algebraic_mul(
#[no_mangle]
pub fn f16_algebraic_mul(a: f16, b: f16) -> f16 {
    // CHECK: fmul reassoc nsz arcp contract half {{(%a, %b)|(%b, %a)}}
    a.algebraic_mul(b)
}

// CHECK-LABEL: half @f16_algebraic_div(
#[no_mangle]
pub fn f16_algebraic_div(a: f16, b: f16) -> f16 {
    // CHECK: fdiv reassoc nsz arcp contract half %a, %b
    a.algebraic_div(b)
}

// CHECK-LABEL: half @f16_algebraic_rem(
#[no_mangle]
pub fn f16_algebraic_rem(a: f16, b: f16) -> f16 {
    // CHECK: frem reassoc nsz arcp contract half %a, %b
    a.algebraic_rem(b)
}
