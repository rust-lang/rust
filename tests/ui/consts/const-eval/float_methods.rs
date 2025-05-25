//@ run-pass
//! Tests the float intrinsics: min, max, abs, copysign

#![feature(f16, f128)]
#![feature(const_float_methods)]

const F16_MIN: f16 = 1.0_f16.min(0.5_f16);
const F16_MAX: f16 = 1.0_f16.max(0.5_f16);
const F16_ABS: f16 = (-1.0_f16).abs();
const F16_COPYSIGN: f16 = 1.0_f16.copysign(-2.0_f16);
const F16_FLOOR: f16 = 0.5_f16.floor();
const F16_CEIL: f16 = 0.5_f16.ceil();
const F16_TRUNC: f16 = 0.5_f16.trunc();
const F16_FRACT: f16 = 0.5_f16.fract();
const F16_ROUND: f16 = 0.5_f16.round();
const F16_ROUND_TIES_EVEN: f16 = 0.5_f16.round_ties_even();

const F32_MIN: f32 = 1.0_f32.min(0.5_f32);
const F32_MAX: f32 = 1.0_f32.max(0.5_f32);
const F32_ABS: f32 = (-1.0_f32).abs();
const F32_COPYSIGN: f32 = 1.0_f32.copysign(-2.0_f32);
const F32_FLOOR: f32 = 0.5_f32.floor();
const F32_CEIL: f32 = 0.5_f32.ceil();
const F32_TRUNC: f32 = 0.5_f32.trunc();
const F32_FRACT: f32 = 0.5_f32.fract();
const F32_ROUND: f32 = 0.5_f32.round();
const F32_ROUND_TIES_EVEN: f32 = 0.5_f32.round_ties_even();

const F64_MIN: f64 = 1.0_f64.min(0.5_f64);
const F64_MAX: f64 = 1.0_f64.max(0.5_f64);
const F64_ABS: f64 = (-1.0_f64).abs();
const F64_COPYSIGN: f64 = 1.0_f64.copysign(-2.0_f64);
const F64_FLOOR: f64 = 0.5_f64.floor();
const F64_CEIL: f64 = 0.5_f64.ceil();
const F64_TRUNC: f64 = 0.5_f64.trunc();
const F64_FRACT: f64 = 0.5_f64.fract();
const F64_ROUND: f64 = 0.5_f64.round();
const F64_ROUND_TIES_EVEN: f64 = 0.5_f64.round_ties_even();

const F128_MIN: f128 = 1.0_f128.min(0.5_f128);
const F128_MAX: f128 = 1.0_f128.max(0.5_f128);
const F128_ABS: f128 = (-1.0_f128).abs();
const F128_COPYSIGN: f128 = 1.0_f128.copysign(-2.0_f128);
const F128_FLOOR: f128 = 0.5_f128.floor();
const F128_CEIL: f128 = 0.5_f128.ceil();
const F128_TRUNC: f128 = 0.5_f128.trunc();
const F128_FRACT: f128 = 0.5_f128.fract();
const F128_ROUND: f128 = 0.5_f128.round();
const F128_ROUND_TIES_EVEN: f128 = 0.5_f128.round_ties_even();

fn main() {
    assert_eq!(F16_MIN, 0.5);
    assert_eq!(F16_MAX, 1.0);
    assert_eq!(F16_ABS, 1.0);
    assert_eq!(F16_COPYSIGN, -1.0);
    assert_eq!(F16_FLOOR, 0.0);
    assert_eq!(F16_CEIL, 1.0);
    assert_eq!(F16_TRUNC, 0.0);
    assert_eq!(F16_FRACT, 0.5);
    assert_eq!(F16_ROUND, 1.0);
    assert_eq!(F16_ROUND_TIES_EVEN, 0.0);

    assert_eq!(F32_MIN, 0.5);
    assert_eq!(F32_MAX, 1.0);
    assert_eq!(F32_ABS, 1.0);
    assert_eq!(F32_COPYSIGN, -1.0);
    assert_eq!(F32_FLOOR, 0.0);
    assert_eq!(F32_CEIL, 1.0);
    assert_eq!(F32_TRUNC, 0.0);
    assert_eq!(F32_FRACT, 0.5);
    assert_eq!(F32_ROUND, 1.0);
    assert_eq!(F32_ROUND_TIES_EVEN, 0.0);

    assert_eq!(F64_MIN, 0.5);
    assert_eq!(F64_MAX, 1.0);
    assert_eq!(F64_ABS, 1.0);
    assert_eq!(F64_COPYSIGN, -1.0);
    assert_eq!(F64_FLOOR, 0.0);
    assert_eq!(F64_CEIL, 1.0);
    assert_eq!(F64_TRUNC, 0.0);
    assert_eq!(F64_FRACT, 0.5);
    assert_eq!(F64_ROUND, 1.0);
    assert_eq!(F64_ROUND_TIES_EVEN, 0.0);

    assert_eq!(F128_MIN, 0.5);
    assert_eq!(F128_MAX, 1.0);
    assert_eq!(F128_ABS, 1.0);
    assert_eq!(F128_COPYSIGN, -1.0);
    assert_eq!(F128_FLOOR, 0.0);
    assert_eq!(F128_CEIL, 1.0);
    assert_eq!(F128_TRUNC, 0.0);
    assert_eq!(F128_FRACT, 0.5);
    assert_eq!(F128_ROUND, 1.0);
    assert_eq!(F128_ROUND_TIES_EVEN, 0.0);
}
