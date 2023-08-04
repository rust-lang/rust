// Verify that our intrinsics generate the correct LLVM calls for f128

#![crate_type = "lib"]
#![feature(f128)]
#![feature(round_ties_even)]
#![feature(core_intrinsics)]

// CHECK-LABEL: fp128 @f128_add(
#[no_mangle]
pub fn f128_add(a: f128, b: f128) -> f128 {
    // CHECK: fadd fp128 %{{.+}}, %{{.+}}
    a + b
}

// CHECK-LABEL: fp128 @f128_sub(
#[no_mangle]
pub fn f128_sub(a: f128, b: f128) -> f128 {
    // CHECK: fsub fp128 %{{.+}}, %{{.+}}
    a - b
}

// CHECK-LABEL: fp128 @f128_mul(
#[no_mangle]
pub fn f128_mul(a: f128, b: f128) -> f128 {
    // CHECK: fmul fp128 %{{.+}}, %{{.+}}
    a * b
}

// CHECK-LABEL: fp128 @f128_div(
#[no_mangle]
pub fn f128_div(a: f128, b: f128) -> f128 {
    // CHECK: fdiv fp128 %{{.+}}, %{{.+}}
    a / b
}

// CHECK-LABEL: fp128 @f128_powi(
#[no_mangle]
pub fn f128_powi(a: f128, n: i32) -> f128 {
    // CHECK: @llvm.powi.f128.i32(fp128 %{{.+}}, i32 %{{.+}})
    a.powi(n)
}

// CHECK-LABEL: fp128 @f128_powf(
#[no_mangle]
pub fn f128_powf(a: f128, b: f128) -> f128 {
    // CHECK: @llvm.pow.f128(fp128 %{{.+}}, fp128 %{{.+}})
    a.powf(b)
}

// CHECK-LABEL: fp128 @f128_sqrt(
#[no_mangle]
pub fn f128_sqrt(a: f128) -> f128 {
    // CHECK: @llvm.sqrt.f128(fp128 %{{.+}})
    a.sqrt()
}

// CHECK-LABEL: fp128 @f128_sin(
#[no_mangle]
pub fn f128_sin(a: f128) -> f128 {
    // CHECK: @llvm.sin.f128(fp128 %{{.+}})
    a.sin()
}

// CHECK-LABEL: fp128 @f128_cos(
#[no_mangle]
pub fn f128_cos(a: f128) -> f128 {
    // CHECK: @llvm.cos.f128(fp128 %{{.+}})
    a.cos()
}

// CHECK-LABEL: fp128 @f128_exp(
#[no_mangle]
pub fn f128_exp(a: f128) -> f128 {
    // CHECK: @llvm.exp.f128(fp128 %{{.+}})
    a.exp()
}

// CHECK-LABEL: fp128 @f128_exp2(
#[no_mangle]
pub fn f128_exp2(a: f128) -> f128 {
    // CHECK: @llvm.exp2.f128(fp128 %{{.+}})
    a.exp2()
}

// CHECK-LABEL: fp128 @f128_log(
#[no_mangle]
pub fn f128_log(a: f128, b: f128) -> f128 {
    // CHECK: @llvm.log.f128(fp128 %{{.+}})
    a.log(b)
}

// CHECK-LABEL: fp128 @f128_log10(
#[no_mangle]
pub fn f128_log10(a: f128) -> f128 {
    // CHECK: @llvm.log10.f128(fp128 %{{.+}})
    a.log10()
}

// CHECK-LABEL: fp128 @f128_log2(
#[no_mangle]
pub fn f128_log2(a: f128) -> f128 {
    // CHECK: @llvm.log2.f128(fp128 %{{.+}})
    a.log2()
}

// CHECK-LABEL: fp128 @f128_fma(
#[no_mangle]
pub fn f128_fma(a: f128, b: f128, c: f128) -> f128 {
    // CHECK: @llvm.fma.f128(fp128 %{{.+}}, fp128 %{{.+}}, fp128 %{{.+}})
    a.mul_add(b, c)
}

// CHECK-LABEL: fp128 @f128_abs(
#[no_mangle]
pub fn f128_abs(a: f128) -> f128 {
    // CHECK: @llvm.fabs.f128(fp128 %{{.+}})
    a.abs()
}

// CHECK-LABEL: fp128 @f128_min(
#[no_mangle]
pub fn f128_min(a: f128, b: f128) -> f128 {
    // CHECK: @llvm.minnum.f128(fp128 %{{.+}}, fp128 %{{.+}})
    a.min(b)
}

// CHECK-LABEL: fp128 @f128_max(
#[no_mangle]
pub fn f128_max(a: f128, b: f128) -> f128 {
    // CHECK: @llvm.maxnum.f128(fp128 %{{.+}}, fp128 %{{.+}})
    a.max(b)
}

// CHECK-LABEL: fp128 @f128_floor(
#[no_mangle]
pub fn f128_floor(a: f128) -> f128 {
    // CHECK: @llvm.floor.f128(fp128 %{{.+}})
    a.floor()
}

// CHECK-LABEL: fp128 @f128_ceil(
#[no_mangle]
pub fn f128_ceil(a: f128) -> f128 {
    // CHECK: @llvm.ceil.f128(fp128 %{{.+}})
    a.ceil()
}

// CHECK-LABEL: fp128 @f128_trunc(
#[no_mangle]
pub fn f128_trunc(a: f128) -> f128 {
    // CHECK: @llvm.trunc.f128(fp128 %{{.+}})
    a.trunc()
}

// CHECK-LABEL: fp128 @f128_copysign(
#[no_mangle]
pub fn f128_copysign(a: f128, b: f128) -> f128 {
    // CHECK: @llvm.copysign.f128(fp128 %{{.+}}, fp128 %{{.+}})
    a.copysign(b)
}

// CHECK-LABEL: fp128 @f128_round(
#[no_mangle]
pub fn f128_round(a: f128) -> f128 {
    // CHECK: @llvm.round.f128(fp128 %{{.+}})
    a.round()
}

// CHECK-LABEL: fp128 @f128_roundeven(
#[no_mangle]
pub fn f128_roundeven(a: f128) -> f128 {
    // CHECK: @llvm.roundeven.f128(fp128 %{{.+}})
    unsafe { std::intrinsics::roundevenf128(a) }
}

// CHECK-LABEL: fp128 @f128_rint(
#[no_mangle]
pub fn f128_rint(a: f128) -> f128 {
    // CHECK: @llvm.rint.f128(fp128 %{{.+}})
    unsafe { std::intrinsics::rintf128(a) }
}
// CHECK-LABEL: fp128 @f128_nearbyint(
#[no_mangle]
pub fn f128_nearbyint(a: f128) -> f128 {
    // CHECK: @llvm.nearbyint.f128(fp128 %{{.+}})
    unsafe { std::intrinsics::nearbyintf128(a) }
}
