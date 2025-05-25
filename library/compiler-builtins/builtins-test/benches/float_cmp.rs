#![cfg_attr(f128_enabled, feature(f128))]

use builtins_test::float_bench;
use compiler_builtins::float::cmp;
use criterion::{Criterion, criterion_main};

/// `gt` symbols are allowed to return differing results, they just get compared
/// to 0.
fn gt_res_eq(a: i32, b: i32) -> bool {
    let a_lt_0 = a <= 0;
    let b_lt_0 = b <= 0;
    (a_lt_0 && b_lt_0) || (!a_lt_0 && !b_lt_0)
}

float_bench! {
    name: cmp_f32_gt,
    sig: (a: f32, b: f32) -> i32,
    crate_fn: cmp::__gtsf2,
    sys_fn: __gtsf2,
    sys_available: all(),
    output_eq: gt_res_eq,
    asm: [
        #[cfg(target_arch = "x86_64")] {
            let ret: i32;
            asm!(
                "xor     {ret:e}, {ret:e}",
                "ucomiss {a}, {b}",
                "seta    {ret:l}",
                a = in(xmm_reg) a,
                b = in(xmm_reg) b,
                ret = out(reg) ret,
                options(nomem, nostack, pure)
            );

            ret
        };

        #[cfg(target_arch = "aarch64")] {
            let ret: i32;
            asm!(
                "fcmp    {a:s}, {b:s}",
                "cset    {ret:w}, gt",
                a = in(vreg) a,
                b = in(vreg) b,
                ret = out(reg) ret,
                options(nomem,nostack),
            );

            ret
        };
    ],
}

float_bench! {
    name: cmp_f32_unord,
    sig: (a: f32, b: f32) -> i32,
    crate_fn: cmp::__unordsf2,
    sys_fn: __unordsf2,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            let ret: i32;
            asm!(
                "xor     {ret:e}, {ret:e}",
                "ucomiss {a}, {b}",
                "setp    {ret:l}",
                a = in(xmm_reg) a,
                b = in(xmm_reg) b,
                ret = out(reg) ret,
                options(nomem, nostack, pure)
            );

            ret
        };

        #[cfg(target_arch = "aarch64")] {
            let ret: i32;
            asm!(
                "fcmp    {a:s}, {b:s}",
                "cset    {ret:w}, vs",
                a = in(vreg) a,
                b = in(vreg) b,
                ret = out(reg) ret,
                options(nomem, nostack, pure)
            );

            ret
        };
    ],
}

float_bench! {
    name: cmp_f64_gt,
    sig: (a: f64, b: f64) -> i32,
    crate_fn: cmp::__gtdf2,
    sys_fn: __gtdf2,
    sys_available: all(),
    output_eq: gt_res_eq,
    asm: [
        #[cfg(target_arch = "x86_64")] {
            let ret: i32;
            asm!(
                "xor     {ret:e}, {ret:e}",
                "ucomisd {a}, {b}",
                "seta    {ret:l}",
                a = in(xmm_reg) a,
                b = in(xmm_reg) b,
                ret = out(reg) ret,
                options(nomem, nostack, pure)
            );

            ret
        };

        #[cfg(target_arch = "aarch64")] {
            let ret: i32;
            asm!(
                "fcmp    {a:d}, {b:d}",
                "cset {ret:w}, gt",
                a = in(vreg) a,
                b = in(vreg) b,
                ret = out(reg) ret,
                options(nomem, nostack, pure)
            );

            ret
        };
    ],
}

float_bench! {
    name: cmp_f64_unord,
    sig: (a: f64, b: f64) -> i32,
    crate_fn: cmp::__unorddf2,
    sys_fn: __unorddf2,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            let ret: i32;
            asm!(
                "xor     {ret:e}, {ret:e}",
                "ucomisd {a}, {b}",
                "setp    {ret:l}",
                a = in(xmm_reg) a,
                b = in(xmm_reg) b,
                ret = out(reg) ret,
                options(nomem, nostack, pure)
            );

            ret
        };

        #[cfg(target_arch = "aarch64")] {
            let ret: i32;
            asm!(
                "fcmp    {a:d}, {b:d}",
                "cset    {ret:w}, vs",
                a = in(vreg) a,
                b = in(vreg) b,
                ret = out(reg) ret,
                options(nomem, nostack, pure)
            );

            ret
        };
    ],
}

float_bench! {
    name: cmp_f128_gt,
    sig: (a: f128, b: f128) -> i32,
    crate_fn: cmp::__gttf2,
    crate_fn_ppc: cmp::__gtkf2,
    sys_fn: __gttf2,
    sys_fn_ppc: __gtkf2,
    sys_available: not(feature = "no-sys-f128"),
    output_eq: gt_res_eq,
    asm: []
}

float_bench! {
    name: cmp_f128_unord,
    sig: (a: f128, b: f128) -> i32,
    crate_fn: cmp::__unordtf2,
    crate_fn_ppc: cmp::__unordkf2,
    sys_fn: __unordtf2,
    sys_fn_ppc: __unordkf2,
    sys_available: not(feature = "no-sys-f128"),
    asm: []
}

pub fn float_cmp() {
    let mut criterion = Criterion::default().configure_from_args();

    cmp_f32_gt(&mut criterion);
    cmp_f32_unord(&mut criterion);
    cmp_f64_gt(&mut criterion);
    cmp_f64_unord(&mut criterion);

    #[cfg(f128_enabled)]
    {
        cmp_f128_gt(&mut criterion);
        cmp_f128_unord(&mut criterion);
    }
}

criterion_main!(float_cmp);
