// Verify that algebraic intrinsics generate the correct LLVM calls for f64

#![crate_type = "lib"]
#![feature(float_algebraic)]

// CHECK-LABEL: double @f64_algebraic_add(
#[no_mangle]
pub fn f64_algebraic_add(a: f64, b: f64) -> f64 {
    // CHECK: fadd reassoc nsz arcp contract double {{(%a, %b)|(%b, %a)}}
    a.algebraic_add(b)
}

// CHECK-LABEL: double @f64_algebraic_sub(
#[no_mangle]
pub fn f64_algebraic_sub(a: f64, b: f64) -> f64 {
    // CHECK: fsub reassoc nsz arcp contract double %a, %b
    a.algebraic_sub(b)
}

// CHECK-LABEL: double @f64_algebraic_mul(
#[no_mangle]
pub fn f64_algebraic_mul(a: f64, b: f64) -> f64 {
    // CHECK: fmul reassoc nsz arcp contract double {{(%a, %b)|(%b, %a)}}
    a.algebraic_mul(b)
}

// CHECK-LABEL: double @f64_algebraic_div(
#[no_mangle]
pub fn f64_algebraic_div(a: f64, b: f64) -> f64 {
    // CHECK: fdiv reassoc nsz arcp contract double %a, %b
    a.algebraic_div(b)
}

// CHECK-LABEL: double @f64_algebraic_rem(
#[no_mangle]
pub fn f64_algebraic_rem(a: f64, b: f64) -> f64 {
    // CHECK: frem reassoc nsz arcp contract double %a, %b
    a.algebraic_rem(b)
}
