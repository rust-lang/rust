// Verify that algebraic intrinsics generate the correct LLVM calls for f128

#![crate_type = "lib"]
#![feature(f128)]
#![feature(float_algebraic)]

// CHECK-LABEL: fp128 @f128_algebraic_add(
#[no_mangle]
pub fn f128_algebraic_add(a: f128, b: f128) -> f128 {
    // CHECK: fadd reassoc nsz arcp contract fp128 {{(%a, %b)|(%b, %a)}}
    a.algebraic_add(b)
}

// CHECK-LABEL: fp128 @f128_algebraic_sub(
#[no_mangle]
pub fn f128_algebraic_sub(a: f128, b: f128) -> f128 {
    // CHECK: fsub reassoc nsz arcp contract fp128 %a, %b
    a.algebraic_sub(b)
}

// CHECK-LABEL: fp128 @f128_algebraic_mul(
#[no_mangle]
pub fn f128_algebraic_mul(a: f128, b: f128) -> f128 {
    // CHECK: fmul reassoc nsz arcp contract fp128 {{(%a, %b)|(%b, %a)}}
    a.algebraic_mul(b)
}

// CHECK-LABEL: fp128 @f128_algebraic_div(
#[no_mangle]
pub fn f128_algebraic_div(a: f128, b: f128) -> f128 {
    // CHECK: fdiv reassoc nsz arcp contract fp128 %a, %b
    a.algebraic_div(b)
}

// CHECK-LABEL: fp128 @f128_algebraic_rem(
#[no_mangle]
pub fn f128_algebraic_rem(a: f128, b: f128) -> f128 {
    // CHECK: frem reassoc nsz arcp contract fp128 %a, %b
    a.algebraic_rem(b)
}
