// By compiling this file we check that all the intrinsics we care about continue to be provided by
// the `compiler_builtins` crate regardless of the changes we make to it. If we, by mistake, stop
// compiling a C implementation and forget to implement that intrinsic in Rust, this file will fail
// to link due to the missing intrinsic (symbol).

#![allow(unused_features)]
#![allow(internal_features)]
#![deny(dead_code)]
#![feature(allocator_api)]
#![feature(f128)]
#![feature(f16)]
#![feature(lang_items)]
#![no_std]
#![no_main]

// Ensure this `compiler_builtins` gets used, rather than the version injected from the sysroot.
extern crate compiler_builtins;
extern crate panic_handler;

// SAFETY: no definitions, only used for linking
#[cfg(all(not(thumb), not(windows), not(target_arch = "wasm32")))]
#[link(name = "c")]
unsafe extern "C" {}

// Every function in this module maps will be lowered to an intrinsic by LLVM, if the platform
// doesn't have native support for the operation used in the function. ARM has a naming convention
// convention for its intrinsics that's different from other architectures; that's why some function
// have an additional comment: the function name is the ARM name for the intrinsic and the comment
// in the non-ARM name for the intrinsic.
mod intrinsics {
    /* f16 operations */

    #[cfg(f16_enabled)]
    pub fn extendhfsf(x: f16) -> f32 {
        x as f32
    }

    #[cfg(f16_enabled)]
    pub fn extendhfdf(x: f16) -> f64 {
        x as f64
    }

    #[cfg(all(f16_enabled, f128_enabled))]
    pub fn extendhftf(x: f16) -> f128 {
        x as f128
    }

    /* f32 operations */

    #[cfg(f16_enabled)]
    pub fn truncsfhf(x: f32) -> f16 {
        x as f16
    }

    // extendsfdf2
    pub fn aeabi_f2d(x: f32) -> f64 {
        x as f64
    }

    #[cfg(f128_enabled)]
    pub fn extendsftf(x: f32) -> f128 {
        x as f128
    }

    // fixsfsi
    pub fn aeabi_f2iz(x: f32) -> i32 {
        x as i32
    }

    // fixsfdi
    pub fn aeabi_f2lz(x: f32) -> i64 {
        x as i64
    }

    pub fn fixsfti(x: f32) -> i128 {
        x as i128
    }

    // fixunssfsi
    pub fn aeabi_f2uiz(x: f32) -> u32 {
        x as u32
    }

    // fixunssfdi
    pub fn aeabi_f2ulz(x: f32) -> u64 {
        x as u64
    }

    pub fn fixunssfti(x: f32) -> u128 {
        x as u128
    }

    // addsf3
    pub fn aeabi_fadd(a: f32, b: f32) -> f32 {
        a + b
    }

    // eqsf2
    pub fn aeabi_fcmpeq(a: f32, b: f32) -> bool {
        a == b
    }

    // gtsf2
    pub fn aeabi_fcmpgt(a: f32, b: f32) -> bool {
        a > b
    }

    // ltsf2
    pub fn aeabi_fcmplt(a: f32, b: f32) -> bool {
        a < b
    }

    // divsf3
    pub fn aeabi_fdiv(a: f32, b: f32) -> f32 {
        a / b
    }

    // mulsf3
    pub fn aeabi_fmul(a: f32, b: f32) -> f32 {
        a * b
    }

    // subsf3
    pub fn aeabi_fsub(a: f32, b: f32) -> f32 {
        a - b
    }

    /* f64 operations */

    // truncdfsf2
    pub fn aeabi_d2f(x: f64) -> f32 {
        x as f32
    }

    // fixdfsi
    pub fn aeabi_d2i(x: f64) -> i32 {
        x as i32
    }

    // fixdfdi
    pub fn aeabi_d2l(x: f64) -> i64 {
        x as i64
    }

    pub fn fixdfti(x: f64) -> i128 {
        x as i128
    }

    // fixunsdfsi
    pub fn aeabi_d2uiz(x: f64) -> u32 {
        x as u32
    }

    // fixunsdfdi
    pub fn aeabi_d2ulz(x: f64) -> u64 {
        x as u64
    }

    pub fn fixunsdfti(x: f64) -> u128 {
        x as u128
    }

    // adddf3
    pub fn aeabi_dadd(a: f64, b: f64) -> f64 {
        a + b
    }

    // eqdf2
    pub fn aeabi_dcmpeq(a: f64, b: f64) -> bool {
        a == b
    }

    // gtdf2
    pub fn aeabi_dcmpgt(a: f64, b: f64) -> bool {
        a > b
    }

    // ltdf2
    pub fn aeabi_dcmplt(a: f64, b: f64) -> bool {
        a < b
    }

    // divdf3
    pub fn aeabi_ddiv(a: f64, b: f64) -> f64 {
        a / b
    }

    // muldf3
    pub fn aeabi_dmul(a: f64, b: f64) -> f64 {
        a * b
    }

    // subdf3
    pub fn aeabi_dsub(a: f64, b: f64) -> f64 {
        a - b
    }

    /* f128 operations */

    #[cfg(all(f16_enabled, f128_enabled))]
    pub fn trunctfhf(x: f128) -> f16 {
        x as f16
    }

    #[cfg(f128_enabled)]
    pub fn trunctfsf(x: f128) -> f32 {
        x as f32
    }

    #[cfg(f128_enabled)]
    pub fn trunctfdf(x: f128) -> f64 {
        x as f64
    }

    #[cfg(f128_enabled)]
    pub fn fixtfsi(x: f128) -> i32 {
        x as i32
    }

    #[cfg(f128_enabled)]
    pub fn fixtfdi(x: f128) -> i64 {
        x as i64
    }

    #[cfg(f128_enabled)]
    pub fn fixtfti(x: f128) -> i128 {
        x as i128
    }

    #[cfg(f128_enabled)]
    pub fn fixunstfsi(x: f128) -> u32 {
        x as u32
    }

    #[cfg(f128_enabled)]
    pub fn fixunstfdi(x: f128) -> u64 {
        x as u64
    }

    #[cfg(f128_enabled)]
    pub fn fixunstfti(x: f128) -> u128 {
        x as u128
    }

    #[cfg(f128_enabled)]
    pub fn addtf(a: f128, b: f128) -> f128 {
        a + b
    }

    #[cfg(f128_enabled)]
    pub fn eqtf(a: f128, b: f128) -> bool {
        a == b
    }

    #[cfg(f128_enabled)]
    pub fn gttf(a: f128, b: f128) -> bool {
        a > b
    }

    #[cfg(f128_enabled)]
    pub fn lttf(a: f128, b: f128) -> bool {
        a < b
    }

    #[cfg(f128_enabled)]
    pub fn multf(a: f128, b: f128) -> f128 {
        a * b
    }

    #[cfg(f128_enabled)]
    pub fn divtf(a: f128, b: f128) -> f128 {
        a / b
    }

    #[cfg(f128_enabled)]
    pub fn subtf(a: f128, b: f128) -> f128 {
        a - b
    }

    /* i32 operations */

    // floatsisf
    pub fn aeabi_i2f(x: i32) -> f32 {
        x as f32
    }

    // floatsidf
    pub fn aeabi_i2d(x: i32) -> f64 {
        x as f64
    }

    #[cfg(f128_enabled)]
    pub fn floatsitf(x: i32) -> f128 {
        x as f128
    }

    pub fn aeabi_idiv(a: i32, b: i32) -> i32 {
        a.wrapping_div(b)
    }

    pub fn aeabi_idivmod(a: i32, b: i32) -> i32 {
        a % b
    }

    /* i64 operations */

    // floatdisf
    pub fn aeabi_l2f(x: i64) -> f32 {
        x as f32
    }

    // floatdidf
    pub fn aeabi_l2d(x: i64) -> f64 {
        x as f64
    }

    #[cfg(f128_enabled)]
    pub fn floatditf(x: i64) -> f128 {
        x as f128
    }

    pub fn mulodi4(a: i64, b: i64) -> i64 {
        a * b
    }

    // divdi3
    pub fn aeabi_ldivmod(a: i64, b: i64) -> i64 {
        a / b
    }

    pub fn moddi3(a: i64, b: i64) -> i64 {
        a % b
    }

    // muldi3
    pub fn aeabi_lmul(a: i64, b: i64) -> i64 {
        a.wrapping_mul(b)
    }

    /* i128 operations */

    pub fn floattisf(x: i128) -> f32 {
        x as f32
    }

    pub fn floattidf(x: i128) -> f64 {
        x as f64
    }

    #[cfg(f128_enabled)]
    pub fn floattitf(x: i128) -> f128 {
        x as f128
    }

    pub fn lshrti3(a: i128, b: usize) -> i128 {
        a >> b
    }

    pub fn divti3(a: i128, b: i128) -> i128 {
        a / b
    }

    pub fn modti3(a: i128, b: i128) -> i128 {
        a % b
    }

    /* u32 operations */

    // floatunsisf
    pub fn aeabi_ui2f(x: u32) -> f32 {
        x as f32
    }

    // floatunsidf
    pub fn aeabi_ui2d(x: u32) -> f64 {
        x as f64
    }

    #[cfg(f128_enabled)]
    pub fn floatunsitf(x: u32) -> f128 {
        x as f128
    }

    pub fn aeabi_uidiv(a: u32, b: u32) -> u32 {
        a / b
    }

    pub fn aeabi_uidivmod(a: u32, b: u32) -> u32 {
        a % b
    }

    /* u64 operations */

    // floatundisf
    pub fn aeabi_ul2f(x: u64) -> f32 {
        x as f32
    }

    // floatundidf
    pub fn aeabi_ul2d(x: u64) -> f64 {
        x as f64
    }

    #[cfg(f128_enabled)]
    pub fn floatunditf(x: u64) -> f128 {
        x as f128
    }

    // udivdi3
    pub fn aeabi_uldivmod(a: u64, b: u64) -> u64 {
        a * b
    }

    pub fn umoddi3(a: u64, b: u64) -> u64 {
        a % b
    }

    /* u128 operations */

    pub fn floatuntisf(x: u128) -> f32 {
        x as f32
    }

    pub fn floatuntidf(x: u128) -> f64 {
        x as f64
    }

    #[cfg(f128_enabled)]
    pub fn floatuntitf(x: u128) -> f128 {
        x as f128
    }

    pub fn muloti4(a: u128, b: u128) -> Option<u128> {
        a.checked_mul(b)
    }

    pub fn multi3(a: u128, b: u128) -> u128 {
        a.wrapping_mul(b)
    }

    pub fn ashlti3(a: u128, b: usize) -> u128 {
        a >> b
    }

    pub fn ashrti3(a: u128, b: usize) -> u128 {
        a << b
    }

    pub fn udivti3(a: u128, b: u128) -> u128 {
        a / b
    }

    pub fn umodti3(a: u128, b: u128) -> u128 {
        a % b
    }
}

fn run() {
    use core::hint::black_box as bb;

    use intrinsics::*;

    // FIXME(f16_f128): some PPC f128 <-> int conversion functions have the wrong names

    #[cfg(f128_enabled)]
    bb(addtf(bb(2.), bb(2.)));
    bb(aeabi_d2f(bb(2.)));
    bb(aeabi_d2i(bb(2.)));
    bb(aeabi_d2l(bb(2.)));
    bb(aeabi_d2uiz(bb(2.)));
    bb(aeabi_d2ulz(bb(2.)));
    bb(aeabi_dadd(bb(2.), bb(3.)));
    bb(aeabi_dcmpeq(bb(2.), bb(3.)));
    bb(aeabi_dcmpgt(bb(2.), bb(3.)));
    bb(aeabi_dcmplt(bb(2.), bb(3.)));
    bb(aeabi_ddiv(bb(2.), bb(3.)));
    bb(aeabi_dmul(bb(2.), bb(3.)));
    bb(aeabi_dsub(bb(2.), bb(3.)));
    bb(aeabi_f2d(bb(2.)));
    bb(aeabi_f2iz(bb(2.)));
    bb(aeabi_f2lz(bb(2.)));
    bb(aeabi_f2uiz(bb(2.)));
    bb(aeabi_f2ulz(bb(2.)));
    bb(aeabi_fadd(bb(2.), bb(3.)));
    bb(aeabi_fcmpeq(bb(2.), bb(3.)));
    bb(aeabi_fcmpgt(bb(2.), bb(3.)));
    bb(aeabi_fcmplt(bb(2.), bb(3.)));
    bb(aeabi_fdiv(bb(2.), bb(3.)));
    bb(aeabi_fmul(bb(2.), bb(3.)));
    bb(aeabi_fsub(bb(2.), bb(3.)));
    bb(aeabi_i2d(bb(2)));
    bb(aeabi_i2f(bb(2)));
    bb(aeabi_idiv(bb(2), bb(3)));
    bb(aeabi_idivmod(bb(2), bb(3)));
    bb(aeabi_l2d(bb(2)));
    bb(aeabi_l2f(bb(2)));
    bb(aeabi_ldivmod(bb(2), bb(3)));
    bb(aeabi_lmul(bb(2), bb(3)));
    bb(aeabi_ui2d(bb(2)));
    bb(aeabi_ui2f(bb(2)));
    bb(aeabi_uidiv(bb(2), bb(3)));
    bb(aeabi_uidivmod(bb(2), bb(3)));
    bb(aeabi_ul2d(bb(2)));
    bb(aeabi_ul2f(bb(2)));
    bb(aeabi_uldivmod(bb(2), bb(3)));
    bb(ashlti3(bb(2), bb(2)));
    bb(ashrti3(bb(2), bb(2)));
    #[cfg(f128_enabled)]
    bb(divtf(bb(2.), bb(2.)));
    bb(divti3(bb(2), bb(2)));
    #[cfg(f128_enabled)]
    bb(eqtf(bb(2.), bb(2.)));
    #[cfg(f16_enabled)]
    bb(extendhfdf(bb(2.)));
    #[cfg(f16_enabled)]
    bb(extendhfsf(bb(2.)));
    #[cfg(all(f16_enabled, f128_enabled))]
    bb(extendhftf(bb(2.)));
    #[cfg(f128_enabled)]
    bb(extendsftf(bb(2.)));
    bb(fixdfti(bb(2.)));
    bb(fixsfti(bb(2.)));
    #[cfg(f128_enabled)]
    bb(fixtfdi(bb(2.)));
    #[cfg(f128_enabled)]
    bb(fixtfsi(bb(2.)));
    #[cfg(f128_enabled)]
    bb(fixtfti(bb(2.)));
    bb(fixunsdfti(bb(2.)));
    bb(fixunssfti(bb(2.)));
    #[cfg(f128_enabled)]
    bb(fixunstfdi(bb(2.)));
    #[cfg(f128_enabled)]
    bb(fixunstfsi(bb(2.)));
    #[cfg(f128_enabled)]
    bb(fixunstfti(bb(2.)));
    #[cfg(f128_enabled)]
    bb(floatditf(bb(2)));
    #[cfg(f128_enabled)]
    bb(floatsitf(bb(2)));
    bb(floattidf(bb(2)));
    bb(floattisf(bb(2)));
    #[cfg(f128_enabled)]
    bb(floattitf(bb(2)));
    #[cfg(f128_enabled)]
    bb(floatunditf(bb(2)));
    #[cfg(f128_enabled)]
    bb(floatunsitf(bb(2)));
    bb(floatuntidf(bb(2)));
    bb(floatuntisf(bb(2)));
    #[cfg(f128_enabled)]
    bb(floatuntitf(bb(2)));
    #[cfg(f128_enabled)]
    bb(gttf(bb(2.), bb(2.)));
    bb(lshrti3(bb(2), bb(2)));
    #[cfg(f128_enabled)]
    bb(lttf(bb(2.), bb(2.)));
    bb(moddi3(bb(2), bb(3)));
    bb(modti3(bb(2), bb(2)));
    bb(mulodi4(bb(2), bb(3)));
    bb(muloti4(bb(2), bb(2)));
    #[cfg(f128_enabled)]
    bb(multf(bb(2.), bb(2.)));
    bb(multi3(bb(2), bb(2)));
    #[cfg(f128_enabled)]
    bb(subtf(bb(2.), bb(2.)));
    #[cfg(f16_enabled)]
    bb(truncsfhf(bb(2.)));
    #[cfg(f128_enabled)]
    bb(trunctfdf(bb(2.)));
    #[cfg(all(f16_enabled, f128_enabled))]
    bb(trunctfhf(bb(2.)));
    #[cfg(f128_enabled)]
    bb(trunctfsf(bb(2.)));
    bb(udivti3(bb(2), bb(2)));
    bb(umoddi3(bb(2), bb(3)));
    bb(umodti3(bb(2), bb(2)));

    something_with_a_dtor(&|| assert_eq!(bb(1), 1));

    // FIXME(#802): This should be re-enabled once a workaround is found.
    // extern "C" {
    //     fn rust_begin_unwind(x: usize);
    // }

    // unsafe {
    //     rust_begin_unwind(0);
    // }
}

fn something_with_a_dtor(f: &dyn Fn()) {
    struct A<'a>(&'a (dyn Fn() + 'a));

    impl Drop for A<'_> {
        fn drop(&mut self) {
            (self.0)();
        }
    }
    let _a = A(f);
    f();
}

#[unsafe(no_mangle)]
#[cfg(not(thumb))]
extern "C" fn main(_argc: core::ffi::c_int, _argv: *const *const u8) -> core::ffi::c_int {
    run();
    0
}

#[unsafe(no_mangle)]
#[cfg(thumb)]
extern "C" fn _start() -> ! {
    run();
    loop {}
}

// SAFETY: no definitions, only used for linking
#[cfg(windows)]
#[link(name = "kernel32")]
#[link(name = "msvcrt")]
unsafe extern "C" {}

// ARM targets need these symbols
#[unsafe(no_mangle)]
pub fn __aeabi_unwind_cpp_pr0() {}

#[unsafe(no_mangle)]
pub fn __aeabi_unwind_cpp_pr1() {}

#[cfg(not(any(windows, target_os = "cygwin")))]
#[allow(non_snake_case)]
#[unsafe(no_mangle)]
pub fn _Unwind_Resume() {}

#[cfg(not(any(windows, target_os = "cygwin")))]
#[lang = "eh_personality"]
pub extern "C" fn eh_personality() {}

#[cfg(any(all(windows, target_env = "gnu"), target_os = "cygwin"))]
mod mingw_unwinding {
    #[unsafe(no_mangle)]
    pub fn rust_eh_personality() {}
    #[unsafe(no_mangle)]
    pub fn rust_eh_unwind_resume() {}
    #[unsafe(no_mangle)]
    pub fn rust_eh_register_frames() {}
    #[unsafe(no_mangle)]
    pub fn rust_eh_unregister_frames() {}
}
