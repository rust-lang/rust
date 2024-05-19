#![feature(f128)]

use compiler_builtins::float::mul;
use criterion::{criterion_group, criterion_main, Criterion};
use testcrate::float_bench;

float_bench! {
    name: mul_f32,
    sig: (a: f32, b: f32) -> f32,
    crate_fn: mul::__mulsf3,
    sys_fn: __mulsf3,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            asm!(
                "mulss {a}, {b}",
                a = inout(xmm_reg) a,
                b = in(xmm_reg) b,
                options(nomem, nostack, pure)
            );

            a
        };

        #[cfg(target_arch = "aarch64")] {
            asm!(
                "fmul {a:s}, {a:s}, {b:s}",
                a = inout(vreg) a,
                b = in(vreg) b,
                options(nomem, nostack, pure)
            );

            a
        };
    ],
}

float_bench! {
    name: mul_f64,
    sig: (a: f64, b: f64) -> f64,
    crate_fn: mul::__muldf3,
    sys_fn: __muldf3,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            asm!(
                "mulsd {a}, {b}",
                a = inout(xmm_reg) a,
                b = in(xmm_reg) b,
                options(nomem, nostack, pure)
            );

            a
        };

        #[cfg(target_arch = "aarch64")] {
            asm!(
                "fmul {a:d}, {a:d}, {b:d}",
                a = inout(vreg) a,
                b = in(vreg) b,
                options(nomem, nostack, pure)
            );

            a
        };
    ],
}

float_bench! {
    name: mul_f128,
    sig: (a: f128, b: f128) -> f128,
    crate_fn: mul::__multf3,
    crate_fn_ppc: mul::__mulkf3,
    sys_fn: __multf3,
    sys_fn_ppc: __mulkf3,
    sys_available: not(feature = "no-sys-f128"),
    asm: []
}

criterion_group!(float_mul, mul_f32, mul_f64, mul_f128);
criterion_main!(float_mul);
