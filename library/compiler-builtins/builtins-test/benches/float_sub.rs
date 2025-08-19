#![cfg_attr(f128_enabled, feature(f128))]

use builtins_test::float_bench;
use compiler_builtins::float::sub;
use criterion::{Criterion, criterion_main};

float_bench! {
    name: sub_f32,
    sig: (a: f32, b: f32) -> f32,
    crate_fn: sub::__subsf3,
    sys_fn: __subsf3,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            asm!(
                "subss {a}, {b}",
                a = inout(xmm_reg) a,
                b = in(xmm_reg) b,
                options(nomem, nostack, pure)
            );

            a
        };

        #[cfg(target_arch = "aarch64")] {
            asm!(
                "fsub {a:s}, {a:s}, {b:s}",
                a = inout(vreg) a,
                b = in(vreg) b,
                options(nomem, nostack, pure)
            );

            a
        };
    ],
}

float_bench! {
    name: sub_f64,
    sig: (a: f64, b: f64) -> f64,
    crate_fn: sub::__subdf3,
    sys_fn: __subdf3,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            asm!(
                "subsd {a}, {b}",
                a = inout(xmm_reg) a,
                b = in(xmm_reg) b,
                options(nomem, nostack, pure)
            );

            a
        };

        #[cfg(target_arch = "aarch64")] {
            asm!(
                "fsub {a:d}, {a:d}, {b:d}",
                a = inout(vreg) a,
                b = in(vreg) b,
                options(nomem, nostack, pure)
            );

            a
        };
    ],
}

#[cfg(f128_enabled)]
float_bench! {
    name: sub_f128,
    sig: (a: f128, b: f128) -> f128,
    crate_fn: sub::__subtf3,
    crate_fn_ppc: sub::__subkf3,
    sys_fn: __subtf3,
    sys_fn_ppc: __subkf3,
    sys_available: not(feature = "no-sys-f128"),
    asm: []
}

pub fn float_sub() {
    let mut criterion = Criterion::default().configure_from_args();

    sub_f32(&mut criterion);
    sub_f64(&mut criterion);

    #[cfg(f128_enabled)]
    {
        sub_f128(&mut criterion);
    }
}

criterion_main!(float_sub);
