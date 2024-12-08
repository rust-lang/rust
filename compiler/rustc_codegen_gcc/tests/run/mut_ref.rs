
// Compiler:
//
// Run-time:
//   stdout: 2
//     7
//     6
//     11

#![allow(internal_features, unused_attributes)]
#![feature(auto_traits, lang_items, no_core, start, intrinsics, rustc_attrs, track_caller)]

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
    #[rustc_nounwind]
    #[rustc_intrinsic]
    #[rustc_intrinsic_must_be_overridden]
    pub fn abort() -> ! {
        loop {}
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

#[track_caller]
#[lang = "panic_const_add_overflow"]
pub fn panic_const_add_overflow() -> ! {
    panic("attempt to add with overflow");
}

/*
 * Code
 */

struct Test {
    field: isize,
}

fn test(num: isize) -> Test {
    Test {
        field: num + 1,
    }
}

fn update_num(num: &mut isize) {
    *num = *num + 5;
}

#[start]
fn main(mut argc: isize, _argv: *const *const u8) -> isize {
    let mut test = test(argc);
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, test.field);
    }
    update_num(&mut test.field);
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, test.field);
    }

    update_num(&mut argc);
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, argc);
    }

    let refe = &mut argc;
    *refe = *refe + 5;
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, argc);
    }

    0
}
