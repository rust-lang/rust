// By compiling this file we check that all the intrinsics we care about continue to be provided by
// the `rustc_builtins` crate regardless of the changes we make to it. If we, by mistake, stop
// compiling a C implementation and forget to implement that intrinsic in Rust, this file will fail
// to link due to the missing intrinsic (symbol).

#![allow(unused_features)]
#![deny(dead_code)]
#![feature(core_float)]
#![feature(lang_items)]
#![feature(libc)]
#![feature(start)]
#![no_std]
#![cfg_attr(thumb, no_main)]

#[cfg(not(thumb))]
extern crate libc;
extern crate rustc_builtins;

// NOTE cfg(not(thumbv6m)) means that the operation is not supported on ARMv6-M at all. Not even
// compiler-rt provides a C/assembly implementation.

// Every function in this module maps will be lowered to an intrinsic by LLVM, if the platform
// doesn't have native support for the operation used in the function. ARM has a naming convention
// convention for its intrinsics that's different from other architectures; that's why some function
// have an additional comment: the function name is the ARM name for the intrinsic and the comment
// in the non-ARM name for the intrinsic.
#[cfg(feature = "c")]
mod intrinsics {
    use core::num::Float;

    // trunccdfsf2
    pub fn aeabi_d2f(x: f64) -> f32 {
        x as f32
    }

    // fixdfsi
    pub fn aeabi_d2i(x: f64) -> i32 {
        x as i32
    }

    // fixdfdi
    #[cfg(not(thumbv6m))]
    pub fn aeabi_d2l(x: f64) -> i64 {
        x as i64
    }

    #[cfg(thumbv6m)]
    pub fn aeabi_d2l(_: f64) -> i64 {
        0
    }

    // fixunsdfsi
    pub fn aeabi_d2uiz(x: f64) -> u32 {
        x as u32
    }

    // fixunsdfdi
    #[cfg(not(thumbv6m))]
    pub fn aeabi_d2ulz(x: f64) -> u64 {
        x as u64
    }

    #[cfg(thumbv6m)]
    pub fn aeabi_d2ulz(_: f64) -> u64 {
        0
    }

    // adddf3
    pub fn aeabi_dadd(a: f64, b: f64) -> f64 {
        a + b
    }

    // eqdf2
    #[cfg(not(thumbv6m))]
    pub fn aeabi_dcmpeq(a: f64, b: f64) -> bool {
        a == b
    }

    #[cfg(thumbv6m)]
    pub fn aeabi_dcmpeq(_: f64, _: f64) -> bool {
        true
    }

    // gtdf2
    #[cfg(not(thumbv6m))]
    pub fn aeabi_dcmpgt(a: f64, b: f64) -> bool {
        a > b
    }

    #[cfg(thumbv6m)]
    pub fn aeabi_dcmpgt(_: f64, _: f64) -> bool {
        true
    }

    // ltdf2
    #[cfg(not(thumbv6m))]
    pub fn aeabi_dcmplt(a: f64, b: f64) -> bool {
        a < b
    }

    #[cfg(thumbv6m)]
    pub fn aeabi_dcmplt(_: f64, _: f64) -> bool {
        true
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

    // extendsfdf2
    pub fn aeabi_f2d(x: f32) -> f64 {
        x as f64
    }

    // fixsfsi
    pub fn aeabi_f2iz(x: f32) -> i32 {
        x as i32
    }

    // fixsfdi
    #[cfg(not(thumbv6m))]
    pub fn aeabi_f2lz(x: f32) -> i64 {
        x as i64
    }

    #[cfg(thumbv6m)]
    pub fn aeabi_f2lz(_: f32) -> i64 {
        0
    }

    // fixunssfsi
    pub fn aeabi_f2uiz(x: f32) -> u32 {
        x as u32
    }

    // fixunssfdi
    #[cfg(not(thumbv6m))]
    pub fn aeabi_f2ulz(x: f32) -> u64 {
        x as u64
    }

    #[cfg(thumbv6m)]
    pub fn aeabi_f2ulz(_: f32) -> u64 {
        0
    }

    // addsf3
    pub fn aeabi_fadd(a: f32, b: f32) -> f32 {
        a + b
    }

    // eqsf2
    #[cfg(not(thumbv6m))]
    pub fn aeabi_fcmpeq(a: f32, b: f32) -> bool {
        a == b
    }

    #[cfg(thumbv6m)]
    pub fn aeabi_fcmpeq(_: f32, _: f32) -> bool {
        true
    }

    // gtsf2
    #[cfg(not(thumbv6m))]
    pub fn aeabi_fcmpgt(a: f32, b: f32) -> bool {
        a > b
    }

    #[cfg(thumbv6m)]
    pub fn aeabi_fcmpgt(_: f32, _: f32) -> bool {
        true
    }

    // ltsf2
    #[cfg(not(thumbv6m))]
    pub fn aeabi_fcmplt(a: f32, b: f32) -> bool {
        a < b
    }

    #[cfg(thumbv6m)]
    pub fn aeabi_fcmplt(_: f32, _: f32) -> bool {
        true
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

    // floatsidf
    pub fn aeabi_i2d(x: i32) -> f64 {
        x as f64
    }

    // floatsisf
    pub fn aeabi_i2f(x: i32) -> f32 {
        x as f32
    }

    pub fn aeabi_idiv(a: i32, b: i32) -> i32 {
        a.wrapping_div(b)
    }

    pub fn aeabi_idivmod(a: i32, b: i32) -> i32 {
        a % b
    }

    // floatdidf
    pub fn aeabi_l2d(x: i64) -> f64 {
        x as f64
    }

    // floatdisf
    pub fn aeabi_l2f(x: i64) -> f32 {
        x as f32
    }

    // divdi3
    pub fn aeabi_ldivmod(a: i64, b: i64) -> i64 {
        a / b
    }

    // muldi3
    pub fn aeabi_lmul(a: i64, b: i64) -> i64 {
        a.wrapping_mul(b)
    }

    // floatunsidf
    pub fn aeabi_ui2d(x: u32) -> f64 {
        x as f64
    }

    // floatunsisf
    pub fn aeabi_ui2f(x: u32) -> f32 {
        x as f32
    }

    pub fn aeabi_uidiv(a: u32, b: u32) -> u32 {
        a / b
    }

    pub fn aeabi_uidivmod(a: u32, b: u32) -> u32 {
        a % b
    }

    // floatundidf
    pub fn aeabi_ul2d(x: u64) -> f64 {
        x as f64
    }

    // floatundisf
    pub fn aeabi_ul2f(x: u64) -> f32 {
        x as f32
    }

    // udivdi3
    pub fn aeabi_uldivmod(a: u64, b: u64) -> u64 {
        a * b
    }

    pub fn moddi3(a: i64, b: i64) -> i64 {
        a % b
    }

    pub fn mulodi4(a: i64, b: i64) -> i64 {
        a * b
    }

    pub fn powidf2(a: f64, b: i32) -> f64 {
        a.powi(b)
    }

    pub fn powisf2(a: f32, b: i32) -> f32 {
        a.powi(b)
    }

    pub fn umoddi3(a: u64, b: u64) -> u64 {
        a % b
    }
}

#[cfg(feature = "c")]
fn run() {
    use core::ptr;
    use intrinsics::*;

    // We use volatile load/stores to prevent LLVM from optimizing away the intrinsics during LTO
    macro_rules! arg {
        () => {
            ptr::read_volatile(0x0 as *const _)
        }
    }

    macro_rules! ret {
        ($e:expr) => {
            unsafe {
                ptr::write_volatile(0x0 as *mut _, $e)
            }
        }
    }

    ret!(aeabi_d2f(arg!()));
    ret!(aeabi_d2i(arg!()));
    ret!(aeabi_d2l(arg!()));
    ret!(aeabi_d2uiz(arg!()));
    ret!(aeabi_d2ulz(arg!()));
    ret!(aeabi_dadd(arg!(), arg!()));
    ret!(aeabi_dcmpeq(arg!(), arg!()));
    ret!(aeabi_dcmpgt(arg!(), arg!()));
    ret!(aeabi_dcmplt(arg!(), arg!()));
    ret!(aeabi_ddiv(arg!(), arg!()));
    ret!(aeabi_dmul(arg!(), arg!()));
    ret!(aeabi_dsub(arg!(), arg!()));
    ret!(aeabi_f2d(arg!()));
    ret!(aeabi_f2iz(arg!()));
    ret!(aeabi_f2lz(arg!()));
    ret!(aeabi_f2uiz(arg!()));
    ret!(aeabi_f2ulz(arg!()));
    ret!(aeabi_fadd(arg!(), arg!()));
    ret!(aeabi_fcmpeq(arg!(), arg!()));
    ret!(aeabi_fcmpgt(arg!(), arg!()));
    ret!(aeabi_fcmplt(arg!(), arg!()));
    ret!(aeabi_fdiv(arg!(), arg!()));
    ret!(aeabi_fmul(arg!(), arg!()));
    ret!(aeabi_fsub(arg!(), arg!()));
    ret!(aeabi_i2d(arg!()));
    ret!(aeabi_i2f(arg!()));
    ret!(aeabi_idiv(arg!(), arg!()));
    ret!(aeabi_idivmod(arg!(), arg!()));
    ret!(aeabi_l2d(arg!()));
    ret!(aeabi_l2f(arg!()));
    ret!(aeabi_ldivmod(arg!(), arg!()));
    ret!(aeabi_lmul(arg!(), arg!()));
    ret!(aeabi_ui2d(arg!()));
    ret!(aeabi_ui2f(arg!()));
    ret!(aeabi_uidiv(arg!(), arg!()));
    ret!(aeabi_uidivmod(arg!(), arg!()));
    ret!(aeabi_ul2d(arg!()));
    ret!(aeabi_ul2f(arg!()));
    ret!(aeabi_uldivmod(arg!(), arg!()));
    ret!(moddi3(arg!(), arg!()));
    ret!(mulodi4(arg!(), arg!()));
    ret!(powidf2(arg!(), arg!()));
    ret!(powisf2(arg!(), arg!()));
    ret!(umoddi3(arg!(), arg!()));
}

#[cfg(all(feature = "c", not(thumb)))]
#[start]
fn main(_: isize, _: *const *const u8) -> isize {
    run();

    0
}

#[cfg(all(not(feature = "c"), not(thumb)))]
#[start]
fn main(_: isize, _: *const *const u8) -> isize {
    0
}

#[cfg(all(feature = "c", thumb))]
#[no_mangle]
pub fn _start() -> ! {
    run();
    loop {}
}

#[cfg(all(not(feature = "c"), thumb))]
#[no_mangle]
pub fn _start() -> ! {
    loop {}
}

// ARM targets need these symbols
#[no_mangle]
pub fn __aeabi_unwind_cpp_pr0() {}

#[no_mangle]
pub fn __aeabi_unwind_cpp_pr1() {}

// Avoid "undefined reference to `_Unwind_Resume`" errors
#[allow(non_snake_case)]
#[no_mangle]
pub fn _Unwind_Resume() {}

// Lang items
#[cfg(not(test))]
#[lang = "eh_personality"]
extern "C" fn eh_personality() {}

#[cfg(not(test))]
#[lang = "panic_fmt"]
extern "C" fn panic_fmt() {}
