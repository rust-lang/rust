#![cfg_attr(f128_enabled, feature(f128))]

use builtins_test::float_bench;
use compiler_builtins::float::pow;
use criterion::{Criterion, criterion_main};

float_bench! {
    name: powi_f32,
    sig: (a: f32, b: i32) -> f32,
    crate_fn: pow::__powisf2,
    sys_fn: __powisf2,
    sys_available: all(),
    asm: [],
}

float_bench! {
    name: powi_f64,
    sig: (a: f64, b: i32) -> f64,
    crate_fn: pow::__powidf2,
    sys_fn: __powidf2,
    sys_available: all(),
    asm: [],
}

// FIXME(f16_f128): can be changed to only `f128_enabled` once `__multf3` and `__divtf3` are
// distributed by nightly.
#[cfg(all(f128_enabled, not(feature = "no-sys-f128")))]
float_bench! {
    name: powi_f128,
    sig: (a: f128, b: i32) -> f128,
    crate_fn: pow::__powitf2,
    crate_fn_ppc: pow::__powikf2,
    sys_fn: __powitf2,
    sys_fn_ppc: __powikf2,
    sys_available: not(feature = "no-sys-f128"),
    asm: []
}

pub fn float_pow() {
    let mut criterion = Criterion::default().configure_from_args();

    powi_f32(&mut criterion);
    powi_f64(&mut criterion);

    #[cfg(all(f128_enabled, not(feature = "no-sys-f128")))]
    powi_f128(&mut criterion);
}

criterion_main!(float_pow);
