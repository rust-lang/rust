//@ run-pass

#![feature(const_float_methods)]
#![feature(f16, f128)]

const F16_MIN: f16 = 1.0_f16.min(0.5_f16);
const F16_MAX: f16 = 1.0_f16.max(0.5_f16);

const F32_MIN: f32 = 1.0_f32.min(0.5_f32);
const F32_MAX: f32 = 1.0_f32.max(0.5_f32);

const F64_MIN: f64 = 1.0_f64.min(0.5_f64);
const F64_MAX: f64 = 1.0_f64.max(0.5_f64);

const F128_MIN: f128 = 1.0_f128.min(0.5_f128);
const F128_MAX: f128 = 1.0_f128.max(0.5_f128);

fn main() {
    assert_eq!(F16_MIN, 0.5);
    assert_eq!(F16_MAX, 1.0);

    assert_eq!(F32_MIN, 0.5);
    assert_eq!(F32_MAX, 1.0);

    assert_eq!(F64_MIN, 0.5);
    assert_eq!(F64_MAX, 1.0);

    assert_eq!(F128_MIN, 0.5);
    assert_eq!(F128_MAX, 1.0);
}
