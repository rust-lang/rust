// By compiling this file we check that all the intrinsics we care about continue to be provided by
// the `compiler_builtins` crate regardless of the changes we make to it. If we, by mistake, stop
// compiling a C implementation and forget to implement that intrinsic in Rust, this file will fail
// to link due to the missing intrinsic (symbol).

#![allow(unused_features)]
#![cfg_attr(thumb, no_main)]
#![deny(dead_code)]
#![feature(alloc_system)]
#![feature(asm)]
#![feature(compiler_builtins_lib)]
#![feature(core_float)]
#![feature(lang_items)]
#![feature(start)]
#![feature(i128_type)]
#![feature(global_allocator)]
#![feature(allocator_api)]
#![cfg_attr(windows, feature(panic_unwind))]
#![no_std]

#[cfg(not(thumb))]
extern crate alloc_system;
extern crate compiler_builtins;
#[cfg(windows)]
extern crate panic_unwind;

#[cfg(not(thumb))]
#[global_allocator]
static A: alloc_system::System = alloc_system::System;

// NOTE cfg(not(thumbv6m)) means that the operation is not supported on ARMv6-M at all. Not even
// compiler-rt provides a C/assembly implementation.

// Every function in this module maps will be lowered to an intrinsic by LLVM, if the platform
// doesn't have native support for the operation used in the function. ARM has a naming convention
// convention for its intrinsics that's different from other architectures; that's why some function
// have an additional comment: the function name is the ARM name for the intrinsic and the comment
// in the non-ARM name for the intrinsic.
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

    pub fn lshrti3(a: i128, b: usize) -> i128 {
        a >> b
    }

    pub fn udivti3(a: u128, b: u128) -> u128 {
        a / b
    }

    pub fn umodti3(a: u128, b: u128) -> u128 {
        a % b
    }

    pub fn divti3(a: i128, b: i128) -> i128 {
        a / b
    }

    pub fn modti3(a: i128, b: i128) -> i128 {
        a % b
    }
}

fn run() {
    use intrinsics::*;

    // A copy of "test::black_box". Used to prevent LLVM from optimizing away the intrinsics during LTO
    fn bb<T>(dummy: T) -> T {
        unsafe { asm!("" : : "r"(&dummy)) }
        dummy
    }

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
    bb(moddi3(bb(2), bb(3)));
    bb(mulodi4(bb(2), bb(3)));
    bb(powidf2(bb(2.), bb(3)));
    bb(powisf2(bb(2.), bb(3)));
    bb(umoddi3(bb(2), bb(3)));
    bb(muloti4(bb(2), bb(2)));
    bb(multi3(bb(2), bb(2)));
    bb(ashlti3(bb(2), bb(2)));
    bb(ashrti3(bb(2), bb(2)));
    bb(lshrti3(bb(2), bb(2)));
    bb(udivti3(bb(2), bb(2)));
    bb(umodti3(bb(2), bb(2)));
    bb(divti3(bb(2), bb(2)));
    bb(modti3(bb(2), bb(2)));

    something_with_a_dtor(&|| assert_eq!(bb(1), 1));
}

fn something_with_a_dtor(f: &Fn()) {
    struct A<'a>(&'a (Fn() + 'a));

    impl<'a> Drop for A<'a> {
        fn drop(&mut self) {
            (self.0)();
        }
    }
    let _a = A(f);
    f();
}

#[cfg(not(thumb))]
#[start]
fn main(_: isize, _: *const *const u8) -> isize {
    run();
    0
}

#[cfg(thumb)]
#[no_mangle]
pub fn _start() -> ! {
    run();
    loop {}
}

#[cfg(windows)]
#[link(name = "kernel32")]
#[link(name = "msvcrt")]
extern {}

// ARM targets need these symbols
#[no_mangle]
pub fn __aeabi_unwind_cpp_pr0() {}

#[no_mangle]
pub fn __aeabi_unwind_cpp_pr1() {}

#[cfg(not(windows))]
#[allow(non_snake_case)]
#[no_mangle]
pub fn _Unwind_Resume() {}

#[cfg(not(windows))]
#[lang = "eh_personality"]
#[no_mangle]
pub extern "C" fn eh_personality() {}

#[lang = "panic_fmt"]
#[no_mangle]
#[allow(private_no_mangle_fns)]
extern "C" fn panic_fmt() {}
