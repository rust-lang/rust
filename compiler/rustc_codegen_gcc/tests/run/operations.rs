// Compiler:
//
// Run-time:
//   stdout: 41
//     39
//     10

#![allow(unused_attributes)]
#![feature(auto_traits, lang_items, no_core, start, intrinsics, arbitrary_self_types)]

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
impl Copy for i16 {}
impl Copy for i32 {}

#[lang = "deref"]
pub trait Deref {
    type Target: ?Sized;

    fn deref(&self) -> &Self::Target;
}

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
        pub fn printf(format: *const i8, ...) -> i32;
        pub fn puts(s: *const u8) -> i32;
        pub fn fflush(stream: *mut i32) -> i32;

        pub static STDOUT: *mut i32;
    }
}

mod intrinsics {
    extern "rust-intrinsic" {
        pub fn abort() -> !;
    }
}

#[lang = "panic"]
#[track_caller]
#[no_mangle]
pub fn panic(_msg: &str) -> ! {
    unsafe {
        libc::puts("Panicking\0" as *const str as *const u8);
        libc::fflush(libc::STDOUT);
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

#[lang = "sub"]
pub trait Sub<RHS = Self> {
    type Output;

    fn sub(self, rhs: RHS) -> Self::Output;
}

impl Sub for usize {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
}

impl Sub for isize {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
}

impl Sub for u8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
}

impl Sub for i8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
}

impl Sub for i16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
}

#[lang = "mul"]
pub trait Mul<RHS = Self> {
    type Output;

    #[must_use]
    fn mul(self, rhs: RHS) -> Self::Output;
}

impl Mul for u8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self * rhs
    }
}

impl Mul for usize {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self * rhs
    }
}

impl Mul for isize {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self * rhs
    }
}

/*
 * Code
 */

#[start]
fn main(mut argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, 40 + argc);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, 40 - argc);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, 10 * argc);
    }
    0
}
