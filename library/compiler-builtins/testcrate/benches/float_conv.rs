#![feature(f128)]
#![allow(improper_ctypes)]

use compiler_builtins::float::conv;
use criterion::{criterion_group, criterion_main, Criterion};
use testcrate::float_bench;

/* unsigned int -> float */

float_bench! {
    name: conv_u32_f32,
    sig: (a: u32) -> f32,
    crate_fn: conv::__floatunsisf,
    sys_fn: __floatunsisf,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            let ret: f32;
            asm!(
                "mov {tmp:e}, {a:e}",
                "cvtsi2ss {ret}, {tmp}",
                a = in(reg) a,
                tmp = out(reg) _,
                ret = lateout(xmm_reg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };

        #[cfg(target_arch = "aarch64")] {
            let ret: f32;
            asm!(
                "ucvtf {ret:s}, {a:w}",
                a = in(reg) a,
                ret = lateout(vreg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

float_bench! {
    name: conv_u32_f64,
    sig: (a: u32) -> f64,
    crate_fn: conv::__floatunsidf,
    sys_fn: __floatunsidf,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            let ret: f64;
            asm!(
                "mov {tmp:e}, {a:e}",
                "cvtsi2sd {ret}, {tmp}",
                a = in(reg) a,
                tmp = out(reg) _,
                ret = lateout(xmm_reg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };

        #[cfg(target_arch = "aarch64")] {
            let ret: f64;
            asm!(
                "ucvtf {ret:d}, {a:w}",
                a = in(reg) a,
                ret = lateout(vreg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

float_bench! {
    name: conv_u64_f32,
    sig: (a: u64) -> f32,
    crate_fn: conv::__floatundisf,
    sys_fn: __floatundisf,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            let ret: f32;
            asm!(
                "ucvtf {ret:s}, {a:x}",
                a = in(reg) a,
                ret = lateout(vreg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

float_bench! {
    name: conv_u64_f64,
    sig: (a: u64) -> f64,
    crate_fn: conv::__floatundidf,
    sys_fn: __floatundidf,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            let ret: f64;
            asm!(
                "ucvtf {ret:d}, {a:x}",
                a = in(reg) a,
                ret = lateout(vreg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

float_bench! {
    name: conv_u128_f32,
    sig: (a: u128) -> f32,
    crate_fn: conv::__floatuntisf,
    sys_fn: __floatuntisf,
    sys_available: all(),
    asm: []
}

float_bench! {
    name: conv_u128_f64,
    sig: (a: u128) -> f64,
    crate_fn: conv::__floatuntidf,
    sys_fn: __floatuntidf,
    sys_available: all(),
    asm: []
}

/* signed int -> float */

float_bench! {
    name: conv_i32_f32,
    sig: (a: i32) -> f32,
    crate_fn: conv::__floatsisf,
    sys_fn: __floatsisf,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            let ret: f32;
            asm!(
                "cvtsi2ss    {ret}, {a:e}",
                a = in(reg) a,
                ret = lateout(xmm_reg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };

        #[cfg(target_arch = "aarch64")] {
            let ret: f32;
            asm!(
                "scvtf {ret:s}, {a:w}",
                a = in(reg) a,
                ret = lateout(vreg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

float_bench! {
    name: conv_i32_f64,
    sig: (a: i32) -> f64,
    crate_fn: conv::__floatsidf,
    sys_fn: __floatsidf,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            let ret: f64;
            asm!(
                "cvtsi2sd    {ret}, {a:e}",
                a = in(reg) a,
                ret = lateout(xmm_reg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };


        #[cfg(target_arch = "aarch64")] {
            let ret: f64;
            asm!(
                "scvtf {ret:d}, {a:w}",
                a = in(reg) a,
                ret = lateout(vreg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

float_bench! {
    name: conv_i64_f32,
    sig: (a: i64) -> f32,
    crate_fn: conv::__floatdisf,
    sys_fn: __floatdisf,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            let ret: f32;
            asm!(
                "cvtsi2ss    {ret}, {a:r}",
                a = in(reg) a,
                ret = lateout(xmm_reg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };

        #[cfg(target_arch = "aarch64")] {
            let ret: f32;
            asm!(
                "scvtf {ret:s}, {a:x}",
                a = in(reg) a,
                ret = lateout(vreg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

float_bench! {
    name: conv_i64_f64,
    sig: (a: i64) -> f64,
    crate_fn: conv::__floatdidf,
    sys_fn: __floatdidf,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "x86_64")] {
            let ret: f64;
            asm!(
                "cvtsi2sd    {ret}, {a:r}",
                a = in(reg) a,
                ret = lateout(xmm_reg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };


        #[cfg(target_arch = "aarch64")] {
            let ret: f64;
            asm!(
                "scvtf {ret:d}, {a:x}",
                a = in(reg) a,
                ret = lateout(vreg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

float_bench! {
    name: conv_i128_f32,
    sig: (a: i128) -> f32,
    crate_fn: conv::__floattisf,
    sys_fn: __floattisf,
    sys_available: all(),
    asm: []
}

float_bench! {
    name: conv_i128_f64,
    sig: (a: i128) -> f64,
    crate_fn: conv::__floattidf,
    sys_fn: __floattidf,
    sys_available: all(),
    asm: []
}

/* float -> unsigned int */

#[cfg(not(all(target_arch = "powerpc64", target_endian = "little")))]
float_bench! {
    name: conv_f32_u32,
    sig: (a: f32) -> u32,
    crate_fn: conv::__fixunssfsi,
    sys_fn: __fixunssfsi,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            let ret: u32;
            asm!(
                "fcvtzu {ret:w}, {a:s}",
                a = in(vreg) a,
                ret = lateout(reg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

#[cfg(not(all(target_arch = "powerpc64", target_endian = "little")))]
float_bench! {
    name: conv_f32_u64,
    sig: (a: f32) -> u64,
    crate_fn: conv::__fixunssfdi,
    sys_fn: __fixunssfdi,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            let ret: u64;
            asm!(
                "fcvtzu {ret:x}, {a:s}",
                a = in(vreg) a,
                ret = lateout(reg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

#[cfg(not(all(target_arch = "powerpc64", target_endian = "little")))]
float_bench! {
    name: conv_f32_u128,
    sig: (a: f32) -> u128,
    crate_fn: conv::__fixunssfti,
    sys_fn: __fixunssfti,
    sys_available: all(),
    asm: []
}

float_bench! {
    name: conv_f64_u32,
    sig: (a: f64) -> u32,
    crate_fn: conv::__fixunsdfsi,
    sys_fn: __fixunsdfsi,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            let ret: u32;
            asm!(
                "fcvtzu {ret:w}, {a:d}",
                a = in(vreg) a,
                ret = lateout(reg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

float_bench! {
    name: conv_f64_u64,
    sig: (a: f64) -> u64,
    crate_fn: conv::__fixunsdfdi,
    sys_fn: __fixunsdfdi,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            let ret: u64;
            asm!(
                "fcvtzu {ret:x}, {a:d}",
                a = in(vreg) a,
                ret = lateout(reg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

float_bench! {
    name: conv_f64_u128,
    sig: (a: f64) -> u128,
    crate_fn: conv::__fixunsdfti,
    sys_fn: __fixunsdfti,
    sys_available: all(),
    asm: []
}

/* float -> signed int */

#[cfg(not(all(target_arch = "powerpc64", target_endian = "little")))]
float_bench! {
    name: conv_f32_i32,
    sig: (a: f32) -> i32,
    crate_fn: conv::__fixsfsi,
    sys_fn: __fixsfsi,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            let ret: i32;
            asm!(
                "fcvtzs {ret:w}, {a:s}",
                a = in(vreg) a,
                ret = lateout(reg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

#[cfg(not(all(target_arch = "powerpc64", target_endian = "little")))]
float_bench! {
    name: conv_f32_i64,
    sig: (a: f32) -> i64,
    crate_fn: conv::__fixsfdi,
    sys_fn: __fixsfdi,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            let ret: i64;
            asm!(
                "fcvtzs {ret:x}, {a:s}",
                a = in(vreg) a,
                ret = lateout(reg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

#[cfg(not(all(target_arch = "powerpc64", target_endian = "little")))]
float_bench! {
    name: conv_f32_i128,
    sig: (a: f32) -> i128,
    crate_fn: conv::__fixsfti,
    sys_fn: __fixsfti,
    sys_available: all(),
    asm: []
}

float_bench! {
    name: conv_f64_i32,
    sig: (a: f64) -> i32,
    crate_fn: conv::__fixdfsi,
    sys_fn: __fixdfsi,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            let ret: i32;
            asm!(
                "fcvtzs {ret:w}, {a:d}",
                a = in(vreg) a,
                ret = lateout(reg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

float_bench! {
    name: conv_f64_i64,
    sig: (a: f64) -> i64,
    crate_fn: conv::__fixdfdi,
    sys_fn: __fixdfdi,
    sys_available: all(),
    asm: [
        #[cfg(target_arch = "aarch64")] {
            let ret: i64;
            asm!(
                "fcvtzs {ret:x}, {a:d}",
                a = in(vreg) a,
                ret = lateout(reg) ret,
                options(nomem, nostack, pure),
            );

            ret
        };
    ],
}

float_bench! {
    name: conv_f64_i128,
    sig: (a: f64) -> i128,
    crate_fn: conv::__fixdfti,
    sys_fn: __fixdfti,
    sys_available: all(),
    asm: []
}

criterion_group!(
    float_conv,
    conv_u32_f32,
    conv_u32_f64,
    conv_u64_f32,
    conv_u64_f64,
    conv_u128_f32,
    conv_u128_f64,
    conv_i32_f32,
    conv_i32_f64,
    conv_i64_f32,
    conv_i64_f64,
    conv_i128_f32,
    conv_i128_f64,
    conv_f64_u32,
    conv_f64_u64,
    conv_f64_u128,
    conv_f64_i32,
    conv_f64_i64,
    conv_f64_i128,
);

// FIXME: ppc64le has a sporadic overflow panic in the crate functions
// <https://github.com/rust-lang/compiler-builtins/issues/617#issuecomment-2125914639>
#[cfg(not(all(target_arch = "powerpc64", target_endian = "little")))]
criterion_group!(
    float_conv_not_ppc64le,
    conv_f32_u32,
    conv_f32_u64,
    conv_f32_u128,
    conv_f32_i32,
    conv_f32_i64,
    conv_f32_i128,
);

#[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
criterion_main!(float_conv);

#[cfg(not(all(target_arch = "powerpc64", target_endian = "little")))]
criterion_main!(float_conv, float_conv_not_ppc64le);
