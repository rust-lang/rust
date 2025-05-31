#![allow(unused_variables)] // "unused" f16 registers
#![cfg_attr(f128_enabled, feature(f128))]
#![cfg_attr(f16_enabled, feature(f16))]

use builtins_test::float_bench;
use compiler_builtins::float::extend;
use criterion::{Criterion, criterion_main};

#[cfg(f16_enabled)]
float_bench! {
    name: extend_f16_f32,
    sig: (a: f16) -> f32,
    crate_fn: extend::__extendhfsf2,
    sys_fn: __extendhfsf2,
    sys_available: not(feature = "no-sys-f16"),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            let ret: f32;
            asm!(
                "fcvt    {ret:s}, {a:h}",
                a = in(vreg) a,
                ret = lateout(vreg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

#[cfg(f16_enabled)]
float_bench! {
    name: extend_f16_f64,
    sig: (a: f16) -> f64,
    crate_fn: extend::__extendhfdf2,
    sys_fn: __extendhfdf2,
    sys_available: not(feature = "no-sys-f16-f64-convert"),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            let ret: f64;
            asm!(
                "fcvt    {ret:d}, {a:h}",
                a = in(vreg) a,
                ret = lateout(vreg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

#[cfg(all(f16_enabled, f128_enabled))]
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

#[cfg(f128_enabled)]
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

#[cfg(f128_enabled)]
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

pub fn float_extend() {
    let mut criterion = Criterion::default().configure_from_args();

    // FIXME(#655): `f16` tests disabled until we can bootstrap symbols
    #[cfg(f16_enabled)]
    #[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64")))]
    {
        extend_f16_f32(&mut criterion);
        extend_f16_f64(&mut criterion);

        #[cfg(f128_enabled)]
        extend_f16_f128(&mut criterion);
    }

    extend_f32_f64(&mut criterion);

    #[cfg(f128_enabled)]
    {
        extend_f32_f128(&mut criterion);
        extend_f64_f128(&mut criterion);
    }
}

criterion_main!(float_extend);
