#![cfg_attr(f128_enabled, feature(f128))]
#![cfg_attr(f16_enabled, feature(f16))]

use builtins_test::float_bench;
use compiler_builtins::float::trunc;
use criterion::{Criterion, criterion_main};

#[cfg(f16_enabled)]
float_bench! {
    name: trunc_f32_f16,
    sig: (a: f32) -> f16,
    crate_fn: trunc::__truncsfhf2,
    sys_fn: __truncsfhf2,
    sys_available: not(feature = "no-sys-f16"),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            let ret: f16;
            asm!(
                "fcvt    {ret:h}, {a:s}",
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
    name: trunc_f64_f16,
    sig: (a: f64) -> f16,
    crate_fn: trunc::__truncdfhf2,
    sys_fn: __truncdfhf2,
    sys_available: not(feature = "no-sys-f16-f64-convert"),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            let ret: f16;
            asm!(
                "fcvt    {ret:h}, {a:d}",
                a = in(vreg) a,
                ret = lateout(vreg) ret,
                options(nomem, nostack, pure),
            );

            ret
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

#[cfg(all(f16_enabled, f128_enabled))]
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

#[cfg(f128_enabled)]
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

#[cfg(f128_enabled)]
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

pub fn float_trunc() {
    let mut criterion = Criterion::default().configure_from_args();

    // FIXME(#655): `f16` tests disabled until we can bootstrap symbols
    #[cfg(f16_enabled)]
    #[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64")))]
    {
        trunc_f32_f16(&mut criterion);
        trunc_f64_f16(&mut criterion);
    }

    trunc_f64_f32(&mut criterion);

    #[cfg(f128_enabled)]
    {
        // FIXME(#655): `f16` tests disabled until we can bootstrap symbols
        #[cfg(f16_enabled)]
        #[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64")))]
        trunc_f128_f16(&mut criterion);

        trunc_f128_f32(&mut criterion);
        trunc_f128_f64(&mut criterion);
    }
}

criterion_main!(float_trunc);
