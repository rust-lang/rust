// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 10
//     10
//     42

#![feature(auto_traits, lang_items, no_core, start, intrinsics)]

#![no_std]
#![no_core]

#[lang = "copy"]
pub unsafe trait Copy {}

unsafe impl Copy for bool {}
unsafe impl Copy for u8 {}
unsafe impl Copy for u16 {}
unsafe impl Copy for u32 {}
unsafe impl Copy for u64 {}
unsafe impl Copy for usize {}
unsafe impl Copy for i8 {}
unsafe impl Copy for i16 {}
unsafe impl Copy for i32 {}
unsafe impl Copy for isize {}
unsafe impl Copy for f32 {}
unsafe impl Copy for char {}

mod libc {
    #[link(name = "c")]
    extern "C" {
        pub fn printf(format: *const i8, ...) -> i32;
    }
}

/*
 * Core
 */

// Because we don't have core yet.
#[lang = "sized"]
pub trait Sized {}

#[lang = "receiver"]
trait Receiver {
}

#[lang = "freeze"]
pub(crate) unsafe auto trait Freeze {}

/*
 * Code
 */

fn int_cast(a: u16, b: i16) -> (u8, u16, u32, usize, i8, i16, i32, isize, u8, u32) {
    (
        a as u8, a as u16, a as u32, a as usize, a as i8, a as i16, a as i32, a as isize, b as u8,
        b as u32,
    )
}

#[start]
fn main(argc: isize, _argv: *const *const u8) -> isize {
    let (a, b, c, d, e, f, g, h, i, j) = int_cast(10, 42);
    unsafe {
        libc::printf(b"%d\n\0" as *const u8 as *const i8, c);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, d);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, j);
    }
    0
}
