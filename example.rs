#![feature(no_core)]
#![no_core]
#![allow(dead_code)]

extern crate mini_core;

use mini_core::*;

fn abc(a: u8) -> u8 {
    a * 2
}

fn bcd(b: bool, a: u8) -> u8 {
    if b {
        a * 2
    } else {
        a * 3
    }
}

// FIXME make calls work
fn call() {
    abc(42);
}

fn indirect_call() {
    let f: fn() = call;
    f();
}

enum BoolOption {
    Some(bool),
    None,
}

fn option_unwrap_or(o: BoolOption, d: bool) -> bool {
    match o {
        BoolOption::Some(b) => b,
        BoolOption::None => d,
    }
}

fn ret_42() -> u8 {
    42
}

fn return_str() -> &'static str {
    "hello world"
}

fn promoted_val() -> &'static u8 {
    &(1 * 2)
}

fn cast_ref_to_raw_ptr(abc: &u8) -> *const u8 {
    abc as *const u8
}

fn cmp_raw_ptr(a: *const u8, b: *const u8) -> bool {
    a == b
}

fn int_cast(a: u16, b: i16) -> (u8, u16, u32, usize, i8, i16, i32, isize, u8, u32) {
    (
        a as u8,
        a as u16,
        a as u32,
        a as usize,
        a as i8,
        a as i16,
        a as i32,
        a as isize,
        b as u8,
        b as u32
    )
}

fn char_cast(c: char) -> u8 {
    c as u8
}

struct DebugTuple(());

fn debug_tuple() -> DebugTuple {
    DebugTuple(())
}

fn size_of<T>() -> usize {
    unsafe {
        intrinsics::size_of::<T>()
    }
}

fn use_size_of() -> usize {
    size_of::<u64>()
}

/*unsafe fn use_copy_intrinsic(src: *const u8, dst: *mut u8) {
    intrinsics::copy::<u8>(src, dst, 1);
}*/

/*unsafe fn use_copy_intrinsic_ref(src: *const u8, dst: *mut u8) {
    let copy2 = &copy::<u8>;
    copy2(src, dst, 1);
}*/

const Abc: u8 = 6 * 7;

fn use_const() -> u8 {
    Abc
}

fn call_closure() {
    (|_, _, _| {

    })(0u8, 42u8, 0u8)
}

fn eq_char(a: char, b: char) -> bool {
    a == b
}

unsafe fn transmute(c: char) -> u32 {
    intrinsics::transmute(c)
}
