// Adapted from https://github.com/sunfishcode/mir2cranelift/blob/master/rust-examples/nocore-hello-world.rs

#![feature(no_core, unboxed_closures, start, lang_items, box_syntax, slice_patterns, never_type, linkage)]
#![no_core]
#![allow(dead_code)]

extern crate mini_core;

use mini_core::*;
use mini_core::libc::*;

unsafe extern "C" fn my_puts(s: *const u8) {
    puts(s);
}

#[lang = "termination"]
trait Termination {
    fn report(self) -> i32;
}

impl Termination for () {
    fn report(self) -> i32 {
        unsafe {
            NUM = 6 * 7 + 1 + (1u8 == 1u8) as u8; // 44
            *NUM_REF as i32
        }
    }
}

trait SomeTrait {
    fn object_safe(&self);
}

impl SomeTrait for &'static str {
    fn object_safe(&self) {
        unsafe {
            puts(*self as *const str as *const u8);
        }
    }
}

struct NoisyDrop {
    text: &'static str,
    inner: NoisyDropInner,
}

struct NoisyDropInner;

impl Drop for NoisyDrop {
    fn drop(&mut self) {
        unsafe {
            puts(self.text as *const str as *const u8);
        }
    }
}

impl Drop for NoisyDropInner {
    fn drop(&mut self) {
        unsafe {
            puts("Inner got dropped!\0" as *const str as *const u8);
        }
    }
}

impl SomeTrait for NoisyDrop {
    fn object_safe(&self) {}
}

enum Ordering {
    Less = -1,
    Equal = 0,
    Greater = 1,
}

#[lang = "start"]
fn start<T: Termination + 'static>(
    main: fn() -> T,
    argc: isize,
    argv: *const *const u8,
) -> isize {
    if argc == 3 {
        unsafe { puts(*argv); }
        unsafe { puts(*((argv as usize + intrinsics::size_of::<*const u8>()) as *const *const u8)); }
        unsafe { puts(*((argv as usize + 2 * intrinsics::size_of::<*const u8>()) as *const *const u8)); }
    }

    main().report();
    0
}

static mut NUM: u8 = 6 * 7;
static NUM_REF: &'static u8 = unsafe { &NUM };

macro_rules! assert {
    ($e:expr) => {
        if !$e {
            panic(&(stringify!(! $e), file!(), line!(), 0));
        }
    };
}

macro_rules! assert_eq {
    ($l:expr, $r: expr) => {
        if $l != $r {
            panic(&(stringify!($l != $r), file!(), line!(), 0));
        }
    }
}

struct Unique<T: ?Sized> {
    pointer: *const T,
    _marker: PhantomData<T>,
}

impl<T: ?Sized, U: ?Sized> CoerceUnsized<Unique<U>> for Unique<T> where T: Unsize<U> {}

fn take_f32(_f: f32) {}
fn take_unique(_u: Unique<()>) {}

fn checked_div_i128(lhs: i128, rhs: i128) -> Option<i128> {
    if rhs == 0 || (lhs == -170141183460469231731687303715884105728 && rhs == -1) {
        None
    } else {
        Some(unsafe { intrinsics::unchecked_div(lhs, rhs) })
    }
}

fn checked_div_u128(lhs: u128, rhs: u128) -> Option<u128> {
    match rhs {
        0 => None,
        rhs => Some(unsafe { intrinsics::unchecked_div(lhs, rhs) })
    }
}

fn main() {
    checked_div_i128(0i128, 2i128);
    checked_div_u128(0u128, 2u128);
}
