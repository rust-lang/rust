#![cfg_attr(f128_enabled, feature(f128))]

use builtins_test::float_bench;
use compiler_builtins::float::conv;
use criterion::{Criterion, criterion_main};

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

#[cfg(f128_enabled)]
float_bench! {
    name: conv_u32_f128,
    sig: (a: u32) -> f128,
    crate_fn: conv::__floatunsitf,
    crate_fn_ppc: conv::__floatunsikf,
    sys_fn: __floatunsitf,
    sys_fn_ppc: __floatunsikf,
    sys_available: not(feature = "no-sys-f16-f128-convert"),
    asm: []
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

#[cfg(f128_enabled)]
float_bench! {
    name: conv_u64_f128,
    sig: (a: u64) -> f128,
    crate_fn: conv::__floatunditf,
    crate_fn_ppc: conv::__floatundikf,
    sys_fn: __floatunditf,
    sys_fn_ppc: __floatundikf,
    sys_available: not(feature = "no-sys-f16-f128-convert"),
    asm: []
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

#[cfg(f128_enabled)]
float_bench! {
    name: conv_u128_f128,
    sig: (a: u128) -> f128,
    crate_fn: conv::__floatuntitf,
    crate_fn_ppc: conv::__floatuntikf,
    sys_fn: __floatuntitf,
    sys_fn_ppc: __floatuntikf,
    sys_available: not(feature = "no-sys-f16-f128-convert"),
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

#[cfg(f128_enabled)]
float_bench! {
    name: conv_i32_f128,
    sig: (a: i32) -> f128,
    crate_fn: conv::__floatsitf,
    crate_fn_ppc: conv::__floatsikf,
    sys_fn: __floatsitf,
    sys_fn_ppc: __floatsikf,
    sys_available: not(feature = "no-sys-f16-f128-convert"),
    asm: []
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

#[cfg(f128_enabled)]
float_bench! {
    name: conv_i64_f128,
    sig: (a: i64) -> f128,
    crate_fn: conv::__floatditf,
    crate_fn_ppc: conv::__floatdikf,
    sys_fn: __floatditf,
    sys_fn_ppc: __floatdikf,
    sys_available: not(feature = "no-sys-f16-f128-convert"),
    asm: []
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

#[cfg(f128_enabled)]
float_bench! {
    name: conv_i128_f128,
    sig: (a: i128) -> f128,
    crate_fn: conv::__floattitf,
    crate_fn_ppc: conv::__floattikf,
    sys_fn: __floattitf,
    sys_fn_ppc: __floattikf,
    sys_available: not(feature = "no-sys-f16-f128-convert"),
    asm: []
}

/* float -> unsigned int */

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

#[cfg(f128_enabled)]
float_bench! {
    name: conv_f128_u32,
    sig: (a: f128) -> u32,
    crate_fn: conv::__fixunstfsi,
    crate_fn_ppc: conv::__fixunskfsi,
    sys_fn: __fixunstfsi,
    sys_available: not(feature = "no-sys-f16-f128-convert"),
    asm: []
}

#[cfg(f128_enabled)]
float_bench! {
    name: conv_f128_u64,
    sig: (a: f128) -> u64,
    crate_fn: conv::__fixunstfdi,
    crate_fn_ppc: conv::__fixunskfdi,
    sys_fn: __fixunstfdi,
    sys_available: not(feature = "no-sys-f16-f128-convert"),
    asm: []
}

#[cfg(f128_enabled)]
float_bench! {
    name: conv_f128_u128,
    sig: (a: f128) -> u128,
    crate_fn: conv::__fixunstfti,
    crate_fn_ppc: conv::__fixunskfti,
    sys_fn: __fixunstfti,
    sys_available: not(feature = "no-sys-f16-f128-convert"),
    asm: []
}

/* float -> signed int */

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

#[cfg(f128_enabled)]
float_bench! {
    name: conv_f128_i32,
    sig: (a: f128) -> i32,
    crate_fn: conv::__fixtfsi,
    crate_fn_ppc: conv::__fixkfsi,
    sys_fn: __fixtfsi,
    sys_available: not(feature = "no-sys-f16-f128-convert"),
    asm: []
}

#[cfg(f128_enabled)]
float_bench! {
    name: conv_f128_i64,
    sig: (a: f128) -> i64,
    crate_fn: conv::__fixtfdi,
    crate_fn_ppc: conv::__fixkfdi,
    sys_fn: __fixtfdi,
    sys_available: not(feature = "no-sys-f16-f128-convert"),
    asm: []
}

#[cfg(f128_enabled)]
float_bench! {
    name: conv_f128_i128,
    sig: (a: f128) -> i128,
    crate_fn: conv::__fixtfti,
    crate_fn_ppc: conv::__fixkfti,
    sys_fn: __fixtfti,
    sys_available: not(feature = "no-sys-f16-f128-convert"),
    asm: []
}

pub fn float_conv() {
    let mut criterion = Criterion::default().configure_from_args();

    conv_u32_f32(&mut criterion);
    conv_u32_f64(&mut criterion);
    conv_u64_f32(&mut criterion);
    conv_u64_f64(&mut criterion);
    conv_u128_f32(&mut criterion);
    conv_u128_f64(&mut criterion);
    conv_i32_f32(&mut criterion);
    conv_i32_f64(&mut criterion);
    conv_i64_f32(&mut criterion);
    conv_i64_f64(&mut criterion);
    conv_i128_f32(&mut criterion);
    conv_i128_f64(&mut criterion);
    conv_f64_u32(&mut criterion);
    conv_f64_u64(&mut criterion);
    conv_f64_u128(&mut criterion);
    conv_f64_i32(&mut criterion);
    conv_f64_i64(&mut criterion);
    conv_f64_i128(&mut criterion);

    #[cfg(f128_enabled)]
    {
        conv_u32_f128(&mut criterion);
        conv_u64_f128(&mut criterion);
        conv_u128_f128(&mut criterion);
        conv_i32_f128(&mut criterion);
        conv_i64_f128(&mut criterion);
        conv_i128_f128(&mut criterion);
        conv_f128_u32(&mut criterion);
        conv_f128_u64(&mut criterion);
        conv_f128_u128(&mut criterion);
        conv_f128_i32(&mut criterion);
        conv_f128_i64(&mut criterion);
        conv_f128_i128(&mut criterion);
    }
}

criterion_main!(float_conv);
