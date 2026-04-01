// Verify that algebraic intrinsics generate the correct LLVM calls

// Ensure operations get inlined
//@ compile-flags: -Copt-level=1

#![crate_type = "lib"]
#![feature(f16)]
#![feature(f128)]
#![feature(float_algebraic)]

// CHECK-LABEL: @f16_algebraic_add
#[no_mangle]
pub fn f16_algebraic_add(a: f16, b: f16) -> f16 {
    // CHECK: fadd reassoc nsz arcp contract half %{{.+}}, %{{.+}}
    a.algebraic_add(b)
}

// CHECK-LABEL: @f16_algebraic_sub
#[no_mangle]
pub fn f16_algebraic_sub(a: f16, b: f16) -> f16 {
    // CHECK: fsub reassoc nsz arcp contract half %{{.+}}, %{{.+}}
    a.algebraic_sub(b)
}

// CHECK-LABEL: @f16_algebraic_mul
#[no_mangle]
pub fn f16_algebraic_mul(a: f16, b: f16) -> f16 {
    // CHECK: fmul reassoc nsz arcp contract half %{{.+}}, %{{.+}}
    a.algebraic_mul(b)
}

// CHECK-LABEL: @f16_algebraic_div
#[no_mangle]
pub fn f16_algebraic_div(a: f16, b: f16) -> f16 {
    // CHECK: fdiv reassoc nsz arcp contract half %{{.+}}, %{{.+}}
    a.algebraic_div(b)
}

// CHECK-LABEL: @f16_algebraic_rem
#[no_mangle]
pub fn f16_algebraic_rem(a: f16, b: f16) -> f16 {
    // CHECK: frem reassoc nsz arcp contract half %{{.+}}, %{{.+}}
    a.algebraic_rem(b)
}

// CHECK-LABEL: @f32_algebraic_add
#[no_mangle]
pub fn f32_algebraic_add(a: f32, b: f32) -> f32 {
    // CHECK: fadd reassoc nsz arcp contract float %{{.+}}, %{{.+}}
    a.algebraic_add(b)
}

// CHECK-LABEL: @f32_algebraic_sub
#[no_mangle]
pub fn f32_algebraic_sub(a: f32, b: f32) -> f32 {
    // CHECK: fsub reassoc nsz arcp contract float %{{.+}}, %{{.+}}
    a.algebraic_sub(b)
}

// CHECK-LABEL: @f32_algebraic_mul
#[no_mangle]
pub fn f32_algebraic_mul(a: f32, b: f32) -> f32 {
    // CHECK: fmul reassoc nsz arcp contract float %{{.+}}, %{{.+}}
    a.algebraic_mul(b)
}

// CHECK-LABEL: @f32_algebraic_div
#[no_mangle]
pub fn f32_algebraic_div(a: f32, b: f32) -> f32 {
    // CHECK: fdiv reassoc nsz arcp contract float %{{.+}}, %{{.+}}
    a.algebraic_div(b)
}

// CHECK-LABEL: @f32_algebraic_rem
#[no_mangle]
pub fn f32_algebraic_rem(a: f32, b: f32) -> f32 {
    // CHECK: frem reassoc nsz arcp contract float %{{.+}}, %{{.+}}
    a.algebraic_rem(b)
}

// CHECK-LABEL: @f64_algebraic_add
#[no_mangle]
pub fn f64_algebraic_add(a: f64, b: f64) -> f64 {
    // CHECK: fadd reassoc nsz arcp contract double %{{.+}}, %{{.+}}
    a.algebraic_add(b)
}

// CHECK-LABEL: @f64_algebraic_sub
#[no_mangle]
pub fn f64_algebraic_sub(a: f64, b: f64) -> f64 {
    // CHECK: fsub reassoc nsz arcp contract double %{{.+}}, %{{.+}}
    a.algebraic_sub(b)
}

// CHECK-LABEL: @f64_algebraic_mul
#[no_mangle]
pub fn f64_algebraic_mul(a: f64, b: f64) -> f64 {
    // CHECK: fmul reassoc nsz arcp contract double %{{.+}}, %{{.+}}
    a.algebraic_mul(b)
}

// CHECK-LABEL: @f64_algebraic_div
#[no_mangle]
pub fn f64_algebraic_div(a: f64, b: f64) -> f64 {
    // CHECK: fdiv reassoc nsz arcp contract double %{{.+}}, %{{.+}}
    a.algebraic_div(b)
}

// CHECK-LABEL: @f64_algebraic_rem
#[no_mangle]
pub fn f64_algebraic_rem(a: f64, b: f64) -> f64 {
    // CHECK: frem reassoc nsz arcp contract double %{{.+}}, %{{.+}}
    a.algebraic_rem(b)
}

// CHECK-LABEL: @f128_algebraic_add
#[no_mangle]
pub fn f128_algebraic_add(a: f128, b: f128) -> f128 {
    // CHECK: fadd reassoc nsz arcp contract fp128 %{{.+}}, %{{.+}}
    a.algebraic_add(b)
}

// CHECK-LABEL: @f128_algebraic_sub
#[no_mangle]
pub fn f128_algebraic_sub(a: f128, b: f128) -> f128 {
    // CHECK: fsub reassoc nsz arcp contract fp128 %{{.+}}, %{{.+}}
    a.algebraic_sub(b)
}

// CHECK-LABEL: @f128_algebraic_mul
#[no_mangle]
pub fn f128_algebraic_mul(a: f128, b: f128) -> f128 {
    // CHECK: fmul reassoc nsz arcp contract fp128 %{{.+}}, %{{.+}}
    a.algebraic_mul(b)
}

// CHECK-LABEL: @f128_algebraic_div
#[no_mangle]
pub fn f128_algebraic_div(a: f128, b: f128) -> f128 {
    // CHECK: fdiv reassoc nsz arcp contract fp128 %{{.+}}, %{{.+}}
    a.algebraic_div(b)
}

// CHECK-LABEL: @f128_algebraic_rem
#[no_mangle]
pub fn f128_algebraic_rem(a: f128, b: f128) -> f128 {
    // CHECK: frem reassoc nsz arcp contract fp128 %{{.+}}, %{{.+}}
    a.algebraic_rem(b)
}
