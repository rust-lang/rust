// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 1
//     2

#![feature(auto_traits, lang_items, no_core, start, intrinsics)]

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

#[lang = "receiver"]
trait Receiver {
}

#[lang = "freeze"]
pub(crate) unsafe auto trait Freeze {}

mod libc {
    #[link(name = "c")]
    extern "C" {
        pub fn printf(format: *const i8, ...) -> i32;
    }
}

/*
 * Code
 */

struct Test {
    field: isize,
}

struct Two {
    two: isize,
}

fn one() -> isize {
    1
}

#[start]
fn main(mut argc: isize, _argv: *const *const u8) -> isize {
    let test = Test {
        field: one(),
    };
    let two = Two {
        two: 2,
    };
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, test.field);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, two.two);
    }
    0
}
