#![allow(unused_variables)] // "unused" f16 registers
#![feature(f128)]
#![feature(f16)]

use compiler_builtins::float::extend;
use criterion::{criterion_group, criterion_main, Criterion};
use testcrate::float_bench;

float_bench! {
    name: extend_f16_f32,
    sig: (a: f16) -> f32,
    crate_fn: extend::__extendhfsf2,
    sys_fn: __extendhfsf2,
    sys_available: not(feature = "no-sys-f16"),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            // FIXME(f16_f128): remove `to_bits()` after f16 asm support (rust-lang/rust/#116909)
            let ret: f32;
            asm!(
                "fcvt    {ret:s}, {a:h}",
                a = in(vreg) a.to_bits(),
                ret = lateout(vreg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

float_bench! {
    name: extend_f16_f128,
    sig: (a: f16) -> f128,
    crate_fn: extend::__extendhftf2,
    crate_fn_ppc: extend::__extendhfkf2,
    sys_fn: __extendhftf2,
    sys_fn_ppc: __extendhfkf2,
    sys_available: not(feature = "no-sys-f16-f128-convert"),
    asm: [],
}

float_bench! {
    name: extend_f32_f64,
    sig: (a: f32) -> f64,
    crate_fn: extend::__extendsfdf2,
    sys_fn: __extendsfdf2,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            let ret: f64;
            asm!(
                "fcvt    {ret:d}, {a:s}",
                a = in(vreg) a,
                ret = lateout(vreg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

float_bench! {
    name: extend_f32_f128,
    sig: (a: f32) -> f128,
    crate_fn: extend::__extendsftf2,
    crate_fn_ppc: extend::__extendsfkf2,
    sys_fn: __extendsftf2,
    sys_fn_ppc: __extendsfkf2,
    sys_available: not(feature = "no-sys-f128"),
    asm: [],
}

float_bench! {
    name: extend_f64_f128,
    sig: (a: f64) -> f128,
    crate_fn: extend::__extenddftf2,
    crate_fn_ppc: extend::__extenddfkf2,
    sys_fn: __extenddftf2,
    sys_fn_ppc: __extenddfkf2,
    sys_available: not(feature = "no-sys-f128"),
    asm: [],
}

criterion_group!(
    float_extend,
    extend_f16_f32,
    extend_f16_f128,
    extend_f32_f64,
    extend_f32_f128,
    extend_f64_f128,
);
criterion_main!(float_extend);
