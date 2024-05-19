#![feature(f128)]
#![feature(f16)]

use compiler_builtins::float::trunc;
use criterion::{criterion_group, criterion_main, Criterion};
use testcrate::float_bench;

float_bench! {
    name: trunc_f32_f16,
    sig: (a: f32) -> f16,
    crate_fn: trunc::__truncsfhf2,
    sys_fn: __truncsfhf2,
    sys_available: not(feature = "no-sys-f16"),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            // FIXME(f16_f128): remove `from_bits()` after f16 asm support (rust-lang/rust/#116909)
            let ret: u16;
            asm!(
                "fcvt    {ret:h}, {a:s}",
                a = in(vreg) a,
                ret = lateout(vreg) ret,
                options(nomem, nostack, pure),
            );

            f16::from_bits(ret)
        };
    ],
}

float_bench! {
    name: trunc_f64_f16,
    sig: (a: f64) -> f16,
    crate_fn: trunc::__truncdfhf2,
    sys_fn: __truncdfhf2,
    sys_available: not(feature = "no-sys-f128"),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            // FIXME(f16_f128): remove `from_bits()` after f16 asm support (rust-lang/rust/#116909)
            let ret: u16;
            asm!(
                "fcvt    {ret:h}, {a:d}",
                a = in(vreg) a,
                ret = lateout(vreg) ret,
                options(nomem, nostack, pure),
            );

            f16::from_bits(ret)
        };
    ],
}

float_bench! {
    name: trunc_f64_f32,
    sig: (a: f64) -> f32,
    crate_fn: trunc::__truncdfsf2,
    sys_fn: __truncdfsf2,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            let ret: f32;
            asm!(
                "cvtsd2ss {ret}, {a}",
                a = in(xmm_reg) a,
                ret = lateout(xmm_reg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };

        #[cfg(target_arch = "aarch64")] {
            let ret: f32;
            asm!(
                "fcvt    {ret:s}, {a:d}",
                a = in(vreg) a,
                ret = lateout(vreg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

float_bench! {
    name: trunc_f128_f16,
    sig: (a: f128) -> f16,
    crate_fn: trunc::__trunctfhf2,
    crate_fn_ppc: trunc::__trunckfhf2,
    sys_fn: __trunctfhf2,
    sys_fn_ppc: __trunckfhf2,
    sys_available: not(feature = "no-sys-f16-f128-convert"),
    asm: [],
}

float_bench! {
    name: trunc_f128_f32,
    sig: (a: f128) -> f32,
    crate_fn: trunc::__trunctfsf2,
    crate_fn_ppc: trunc::__trunckfsf2,
    sys_fn: __trunctfsf2,
    sys_fn_ppc: __trunckfsf2,
    sys_available: not(feature = "no-sys-f128"),
    asm: [],
}

float_bench! {
    name: trunc_f128_f64,
    sig: (a: f128) -> f64,
    crate_fn: trunc::__trunctfdf2,
    crate_fn_ppc: trunc::__trunckfdf2,
    sys_fn: __trunctfdf2,
    sys_fn_ppc: __trunckfdf2,
    sys_available: not(feature = "no-sys-f128"),
    asm: [],
}

criterion_group!(
    float_trunc,
    trunc_f32_f16,
    trunc_f64_f16,
    trunc_f64_f32,
    trunc_f128_f16,
    trunc_f128_f32,
    trunc_f128_f64,
);
criterion_main!(float_trunc);
