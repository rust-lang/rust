//@aux-build:proc_macros.rs:proc-macro
#![allow(unused)]
#![allow(deref_nullptr)]
#![allow(clippy::unnecessary_operation)]
#![allow(dropping_copy_types)]
#![warn(clippy::multiple_unsafe_ops_per_block)]

extern crate proc_macros;
use proc_macros::external;

use core::arch::asm;

fn raw_ptr() -> *const () {
    core::ptr::null()
}

unsafe fn not_very_safe() {}

struct Sample;

impl Sample {
    unsafe fn not_very_safe(&self) {}
}

#[allow(non_upper_case_globals)]
const sample: Sample = Sample;

union U {
    i: i32,
    u: u32,
}

static mut STATIC: i32 = 0;

fn test1() {
    unsafe {
        STATIC += 1;
        not_very_safe();
    }
}

fn test2() {
    let u = U { i: 0 };

    unsafe {
        drop(u.u);
        *raw_ptr();
    }
}

fn test3() {
    unsafe {
        asm!("nop");
        sample.not_very_safe();
        STATIC = 0;
    }
}

fn test_all() {
    let u = U { i: 0 };
    unsafe {
        drop(u.u);
        drop(STATIC);
        sample.not_very_safe();
        not_very_safe();
        *raw_ptr();
        asm!("nop");
    }
}

// no lint
fn correct1() {
    unsafe {
        STATIC += 1;
    }
}

// no lint
fn correct2() {
    unsafe {
        STATIC += 1;
    }

    unsafe {
        *raw_ptr();
    }
}

// no lint
fn correct3() {
    let u = U { u: 0 };

    unsafe {
        not_very_safe();
    }

    unsafe {
        drop(u.i);
    }
}

// tests from the issue (https://github.com/rust-lang/rust-clippy/issues/10064)

unsafe fn read_char_bad(ptr: *const u8) -> char {
    unsafe { char::from_u32_unchecked(*ptr.cast::<u32>()) }
}

// no lint
unsafe fn read_char_good(ptr: *const u8) -> char {
    let int_value = unsafe { *ptr.cast::<u32>() };
    unsafe { core::char::from_u32_unchecked(int_value) }
}

// no lint
fn issue10259() {
    external!(unsafe {
        *core::ptr::null::<()>();
        *core::ptr::null::<()>();
    });
}

fn _fn_ptr(x: unsafe fn()) {
    unsafe {
        x();
        x();
    }
}

fn _assoc_const() {
    trait X {
        const X: unsafe fn();
    }
    fn _f<T: X>() {
        unsafe {
            T::X();
            T::X();
        }
    }
}

fn _field_fn_ptr(x: unsafe fn()) {
    struct X(unsafe fn());
    let x = X(x);
    unsafe {
        x.0();
        x.0();
    }
}

fn main() {}
