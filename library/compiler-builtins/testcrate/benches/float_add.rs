#![feature(f128)]

use compiler_builtins::float::add;
use criterion::{criterion_group, criterion_main, Criterion};
use testcrate::float_bench;

float_bench! {
    name: add_f32,
    sig: (a: f32, b: f32) -> f32,
    crate_fn: add::__addsf3,
    sys_fn: __addsf3,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            asm!(
                "addss {a}, {b}",
                a = inout(xmm_reg) a,
                b = in(xmm_reg) b,
                options(nomem, nostack, pure)
            );

            a
        };

        #[cfg(target_arch = "aarch64")] {
            asm!(
                "fadd {a:s}, {a:s}, {b:s}",
                a = inout(vreg) a,
                b = in(vreg) b,
                options(nomem, nostack, pure)
            );

            a
        };
    ],
}

float_bench! {
    name: add_f64,
    sig: (a: f64, b: f64) -> f64,
    crate_fn: add::__adddf3,
    sys_fn: __adddf3,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            asm!(
                "addsd {a}, {b}",
                a = inout(xmm_reg) a,
                b = in(xmm_reg) b,
                options(nomem, nostack, pure)
            );

            a
        };

        #[cfg(target_arch = "aarch64")] {
            asm!(
                "fadd {a:d}, {a:d}, {b:d}",
                a = inout(vreg) a,
                b = in(vreg) b,
                options(nomem, nostack, pure)
            );

            a
        };
    ],
}

float_bench! {
    name: add_f128,
    sig: (a: f128, b: f128) -> f128,
    crate_fn: add::__addtf3,
    crate_fn_ppc: add::__addkf3,
    sys_fn: __addtf3,
    sys_fn_ppc: __addkf3,
    sys_available: not(feature = "no-sys-f128"),
    asm: []
}

criterion_group!(float_add, add_f32, add_f64, add_f128);
criterion_main!(float_add);
