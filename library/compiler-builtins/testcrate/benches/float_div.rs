#![feature(f128)]

use compiler_builtins::float::div;
use criterion::{criterion_group, criterion_main, Criterion};
use testcrate::float_bench;

float_bench! {
    name: div_f32,
    sig: (a: f32, b: f32) -> f32,
    crate_fn: div::__divsf3,
    sys_fn: __divsf3,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            asm!(
                "divss {a}, {b}",
                a = inout(xmm_reg) a,
                b = in(xmm_reg) b,
                options(nomem, nostack, pure)
            );

            a
        };

        #[cfg(target_arch = "aarch64")] {
            asm!(
                "fdiv {a:s}, {a:s}, {b:s}",
                a = inout(vreg) a,
                b = in(vreg) b,
                options(nomem, nostack, pure)
            );

            a
        };
    ],
}

float_bench! {
    name: div_f64,
    sig: (a: f64, b: f64) -> f64,
    crate_fn: div::__divdf3,
    sys_fn: __divdf3,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            asm!(
                "divsd {a}, {b}",
                a = inout(xmm_reg) a,
                b = in(xmm_reg) b,
                options(nomem, nostack, pure)
            );

            a
        };

        #[cfg(target_arch = "aarch64")] {
            asm!(
                "fdiv {a:d}, {a:d}, {b:d}",
                a = inout(vreg) a,
                b = in(vreg) b,
                options(nomem, nostack, pure)
            );

            a
        };
    ],
}

criterion_group!(float_div, div_f32, div_f64);
criterion_main!(float_div);
