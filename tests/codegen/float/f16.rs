// Verify that our intrinsics generate the correct LLVM calls for f16

#![crate_type = "lib"]
#![feature(f16)]
#![feature(round_ties_even)]
#![feature(core_intrinsics)]

// CHECK-LABEL: half @f16_add(
#[no_mangle]
pub fn f16_add(a: f16, b: f16) -> f16 {
    // CHECK: fadd half %{{.+}}, %{{.+}}
    a + b
}

// CHECK-LABEL: half @f16_sub(
#[no_mangle]
pub fn f16_sub(a: f16, b: f16) -> f16 {
    // CHECK: fsub half %{{.+}}, %{{.+}}
    a - b
}

// CHECK-LABEL: half @f16_mul(
#[no_mangle]
pub fn f16_mul(a: f16, b: f16) -> f16 {
    // CHECK: fmul half %{{.+}}, %{{.+}}
    a * b
}

// CHECK-LABEL: half @f16_div(
#[no_mangle]
pub fn f16_div(a: f16, b: f16) -> f16 {
    // CHECK: fdiv half %{{.+}}, %{{.+}}
    a / b
}

// CHECK-LABEL: half @f16_powi(
#[no_mangle]
pub fn f16_powi(a: f16, n: i32) -> f16 {
    // CHECK: @llvm.powi.f16.i32(half %{{.+}}, i32 %{{.+}})
    a.powi(n)
}

// CHECK-LABEL: half @f16_powf(
#[no_mangle]
pub fn f16_powf(a: f16, b: f16) -> f16 {
    // CHECK: @llvm.pow.f16(half %{{.+}}, half %{{.+}})
    a.powf(b)
}

// CHECK-LABEL: half @f16_sqrt(
#[no_mangle]
pub fn f16_sqrt(a: f16) -> f16 {
    // CHECK: @llvm.sqrt.f16(half %{{.+}})
    a.sqrt()
}

// CHECK-LABEL: half @f16_sin(
#[no_mangle]
pub fn f16_sin(a: f16) -> f16 {
    // CHECK: @llvm.sin.f16(half %{{.+}})
    a.sin()
}

// CHECK-LABEL: half @f16_cos(
#[no_mangle]
pub fn f16_cos(a: f16) -> f16 {
    // CHECK: @llvm.cos.f16(half %{{.+}})
    a.cos()
}

// CHECK-LABEL: half @f16_exp(
#[no_mangle]
pub fn f16_exp(a: f16) -> f16 {
    // CHECK: @llvm.exp.f16(half %{{.+}})
    a.exp()
}

// CHECK-LABEL: half @f16_exp2(
#[no_mangle]
pub fn f16_exp2(a: f16) -> f16 {
    // CHECK: @llvm.exp2.f16(half %{{.+}})
    a.exp2()
}

// CHECK-LABEL: half @f16_log(
#[no_mangle]
pub fn f16_log(a: f16, b: f16) -> f16 {
    // CHECK: @llvm.log.f16(half %{{.+}})
    a.log(b)
}

// CHECK-LABEL: half @f16_log10(
#[no_mangle]
pub fn f16_log10(a: f16) -> f16 {
    // CHECK: @llvm.log10.f16(half %{{.+}})
    a.log10()
}

// CHECK-LABEL: half @f16_log2(
#[no_mangle]
pub fn f16_log2(a: f16) -> f16 {
    // CHECK: @llvm.log2.f16(half %{{.+}})
    a.log2()
}

// CHECK-LABEL: half @f16_fma(
#[no_mangle]
pub fn f16_fma(a: f16, b: f16, c: f16) -> f16 {
    // CHECK: @llvm.fma.f16(half %{{.+}}, half %{{.+}}, half %{{.+}})
    a.mul_add(b, c)
}

// CHECK-LABEL: half @f16_abs(
#[no_mangle]
pub fn f16_abs(a: f16) -> f16 {
    // CHECK: @llvm.fabs.f16(half %{{.+}})
    a.abs()
}

// CHECK-LABEL: half @f16_min(
#[no_mangle]
pub fn f16_min(a: f16, b: f16) -> f16 {
    // CHECK: @llvm.minnum.f16(half %{{.+}}, half %{{.+}})
    a.min(b)
}

// CHECK-LABEL: half @f16_max(
#[no_mangle]
pub fn f16_max(a: f16, b: f16) -> f16 {
    // CHECK: @llvm.maxnum.f16(half %{{.+}}, half %{{.+}})
    a.max(b)
}

// CHECK-LABEL: half @f16_floor(
#[no_mangle]
pub fn f16_floor(a: f16) -> f16 {
    // CHECK: @llvm.floor.f16(half %{{.+}})
    a.floor()
}

// CHECK-LABEL: half @f16_ceil(
#[no_mangle]
pub fn f16_ceil(a: f16) -> f16 {
    // CHECK: @llvm.ceil.f16(half %{{.+}})
    a.ceil()
}

// CHECK-LABEL: half @f16_trunc(
#[no_mangle]
pub fn f16_trunc(a: f16) -> f16 {
    // CHECK: @llvm.trunc.f16(half %{{.+}})
    a.trunc()
}

// CHECK-LABEL: half @f16_copysign(
#[no_mangle]
pub fn f16_copysign(a: f16, b: f16) -> f16 {
    // CHECK: @llvm.copysign.f16(half %{{.+}}, half %{{.+}})
    a.copysign(b)
}

// CHECK-LABEL: half @f16_round(
#[no_mangle]
pub fn f16_round(a: f16) -> f16 {
    // CHECK: @llvm.round.f16(half %{{.+}})
    a.round()
}

// CHECK-LABEL: half @f16_roundeven(
#[no_mangle]
pub fn f16_roundeven(a: f16) -> f16 {
    // CHECK: @llvm.roundeven.f16(half %{{.+}})
    unsafe { std::intrinsics::roundevenf16(a) }
}

// CHECK-LABEL: half @f16_rint(
#[no_mangle]
pub fn f16_rint(a: f16) -> f16 {
    // CHECK: @llvm.rint.f16(half %{{.+}})
    unsafe { std::intrinsics::rintf16(a) }
}

// CHECK-LABEL: half @f16_nearbyint(
#[no_mangle]
pub fn f16_nearbyint(a: f16) -> f16 {
    // CHECK: @llvm.nearbyint.f16(half %{{.+}})
    unsafe { std::intrinsics::nearbyintf16(a) }
}
