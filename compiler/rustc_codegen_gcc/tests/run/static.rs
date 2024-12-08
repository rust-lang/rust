// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 10
//      14
//      1
//      12
//      12
//      1

#![feature(auto_traits, lang_items, no_core, start, intrinsics, rustc_attrs)]
#![allow(internal_features)]

#![no_std]
#![no_core]

/*
 * Core
 */

// Because we don't have core yet.
#[lang = "sized"]
pub trait Sized {}

#[lang = "destruct"]
pub trait Destruct {}

#[lang = "drop"]
pub trait Drop {}

#[lang = "copy"]
trait Copy {
}

impl Copy for isize {}
impl<T: ?Sized> Copy for *mut T {}

#[lang = "receiver"]
trait Receiver {
}

#[lang = "freeze"]
pub(crate) unsafe auto trait Freeze {}

mod intrinsics {
    use super::Sized;

    #[rustc_nounwind]
    #[rustc_intrinsic]
    #[rustc_intrinsic_must_be_overridden]
    pub fn abort() -> ! {
        loop {}
    }
}

mod libc {
    #[link(name = "c")]
    extern "C" {
        pub fn printf(format: *const i8, ...) -> i32;
    }
}

#[lang = "structural_peq"]
pub trait StructuralPartialEq {}

#[lang = "drop_in_place"]
#[allow(unconditional_recursion)]
pub unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    // Code here does not matter - this is replaced by the
    // real drop glue by the compiler.
    drop_in_place(to_drop);
}

/*
 * Code
 */

struct Test {
    field: isize,
}

struct WithRef {
    refe: &'static Test,
}

static mut CONSTANT: isize = 10;

static mut TEST: Test = Test {
    field: 12,
};

static mut TEST2: Test = Test {
    field: 14,
};

static mut WITH_REF: WithRef = WithRef {
    refe: unsafe { &TEST },
};

#[start]
fn main(mut argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, CONSTANT);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, TEST2.field);
        TEST2.field = argc;
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, TEST2.field);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, WITH_REF.refe.field);
        WITH_REF.refe = &TEST2;
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, TEST.field);
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, WITH_REF.refe.field);
    }
    0
}
