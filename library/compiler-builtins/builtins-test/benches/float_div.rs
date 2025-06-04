#![cfg_attr(f128_enabled, feature(f128))]

use builtins_test::float_bench;
use compiler_builtins::float::div;
use criterion::{Criterion, criterion_main};

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

#[cfg(f128_enabled)]
float_bench! {
    name: div_f128,
    sig: (a: f128, b: f128) -> f128,
    crate_fn: div::__divtf3,
    crate_fn_ppc: div::__divkf3,
    sys_fn: __divtf3,
    sys_fn_ppc: __divkf3,
    sys_available: not(feature = "no-sys-f128"),
    asm: []
}

pub fn float_div() {
    let mut criterion = Criterion::default().configure_from_args();

    div_f32(&mut criterion);
    div_f64(&mut criterion);

    #[cfg(f128_enabled)]
    {
        div_f128(&mut criterion);
    }
}

criterion_main!(float_div);
