// Compiler:
//
// Run-time:
//   stdout: 2
//     7 8
//     10

#![allow(unused_attributes)]
#![feature(auto_traits, lang_items, no_core, start, intrinsics, track_caller)]

#![no_std]
#![no_core]

/*
 * Core
 */

// Because we don't have core yet.
#[lang = "sized"]
pub trait Sized {}

#[lang = "copy"]
trait Copy {
}

impl Copy for isize {}
impl Copy for *mut i32 {}
impl Copy for usize {}
impl Copy for u8 {}
impl Copy for i8 {}
impl Copy for i32 {}

#[lang = "receiver"]
trait Receiver {
}

#[lang = "freeze"]
pub(crate) unsafe auto trait Freeze {}

#[lang = "panic_location"]
struct PanicLocation {
    file: &'static str,
    line: u32,
    column: u32,
}

mod libc {
    #[link(name = "c")]
    extern "C" {
        pub fn puts(s: *const u8) -> i32;
        pub fn fflush(stream: *mut i32) -> i32;
        pub fn printf(format: *const i8, ...) -> i32;

        pub static stdout: *mut i32;
    }
}

mod intrinsics {
    extern "rust-intrinsic" {
        #[rustc_safe_intrinsic]
        pub fn abort() -> !;
    }
}

#[lang = "panic"]
#[track_caller]
#[no_mangle]
pub fn panic(_msg: &'static str) -> ! {
    unsafe {
        libc::puts("Panicking\0" as *const str as *const u8);
        libc::fflush(libc::stdout);
        intrinsics::abort();
    }
}

#[lang = "add"]
trait Add<RHS = Self> {
    type Output;

    fn add(self, rhs: RHS) -> Self::Output;
}

impl Add for u8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl Add for i8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl Add for i32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl Add for usize {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl Add for isize {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

/*
 * Code
 */

fn inc_ref(num: &mut isize) -> isize {
    *num = *num + 5;
    *num + 1
}

fn inc(num: isize) -> isize {
    num + 1
}


#[start]
fn main(mut argc: isize, _argv: *const *const u8) -> isize {
    argc = inc(argc);
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, argc);
    }

    let b = inc_ref(&mut argc);
    unsafe {
        libc::printf(b"%ld %ld\n\0" as *const u8 as *const i8, argc, b);
    }

    argc = 10;
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, argc);
    }
    0
}
