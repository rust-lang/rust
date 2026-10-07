// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 7 8 9
//     3 2 1
//     4 5 6
//     10 11 12
//     20 21 22

#![feature(no_core)]
#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;
use mini_core::*;

// 24 bytes: returned indirectly (sret) under both the Rust and the C ABI.
#[repr(C)]
struct Big {
    a: u64,
    b: u64,
    c: u64,
}

// 12 bytes: returned indirectly under the Rust ABI only (the C ABI would
// return it in registers), so this exercises the forced memory return.
struct Mid {
    a: u32,
    b: u32,
    c: u32,
}

struct Pair {
    first: u64,
    second: Big,
}

#[inline(never)]
fn make_big(a: u64, b: u64, c: u64) -> Big {
    Big { a, b, c }
}

#[inline(never)]
extern "C" fn make_big_c(a: u64, b: u64, c: u64) -> Big {
    Big { a, b, c }
}

#[inline(never)]
fn make_mid(a: u32, b: u32, c: u32) -> Mid {
    Mid { a, b, c }
}

// The shape that used to fail in getopts: a 24-byte tuple.
#[inline(never)]
fn make_tuple(a: usize, b: u64, c: u64) -> (usize, u64, u64) {
    (a, b, c)
}

extern "C" {
    // A GCC-built callee for a cg_gcc caller, and a GCC-built caller of `rust_make_big`.
    fn c_make_big(a: i64, b: i64, c: i64) -> Big;
    fn c_call_rust() -> i32;
}

// The callee for the GCC-built caller in `c_call_rust`. `#[no_mangle]` also pins the calling
// convention: with internal linkage, GCC may clone the function with a changed convention at
// `-O3` and the arguments never travel through the ABI-mandated registers.
#[no_mangle]
extern "C" fn rust_make_big(a: i64, b: i64, c: i64) -> Big {
    Big { a: a as u64, b: b as u64, c: c as u64 }
}

// On x86-64 the hidden pointer is passed in the first argument register, which is also where an
// ordinary first parameter goes, so the C round-trip cannot tell the two apart there. The
// other half of the convention can: the callee must hand the hidden pointer back in the return
// register. A backend that declares the return pointer as an explicit parameter and returns void
// fills the struct but leaves the return register undefined.
#[cfg(target_arch = "x86_64")]
unsafe fn check_c_return_register() {
    // The function pointer is made opaque so that GCC cannot recover the callee declaration and
    // classify the call from it instead of from the pointer type.
    let as_c_caller: extern "C" fn(*mut Big, u64, u64, u64) -> *mut Big = intrinsics::black_box(
        intrinsics::transmute(make_big_c as extern "C" fn(u64, u64, u64) -> Big),
    );
    let mut out = Big { a: 0, b: 0, c: 0 };
    let returned = as_c_caller(&mut out, 40, 41, 42);
    if returned as usize != &mut out as *mut Big as usize
        || out.a != 40
        || out.b != 41
        || out.c != 42
    {
        intrinsics::abort();
    }
}

#[no_mangle]
extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    unsafe {
        let big = make_big(7, 8, 9);
        libc::printf(b"%llu %llu %llu\n\0" as *const u8 as *const i8, big.a, big.b, big.c);

        let big_c = make_big_c(3, 2, 1);
        libc::printf(b"%llu %llu %llu\n\0" as *const u8 as *const i8, big_c.a, big_c.b, big_c.c);

        let mid = make_mid(4, 5, 6);
        libc::printf(b"%d %d %d\n\0" as *const u8 as *const i8, mid.a, mid.b, mid.c);

        let tuple = make_tuple(10, 11, 12);
        libc::printf(b"%ld %llu %llu\n\0" as *const u8 as *const i8, tuple.0, tuple.1, tuple.2);

        // Store an indirect return value into a field of a local.
        let mut pair = Pair { first: 0, second: make_big(1, 2, 3) };
        pair.second = make_big(20, 21, 22);
        libc::printf(
            b"%llu %llu %llu\n\0" as *const u8 as *const i8,
            pair.second.a,
            pair.second.b,
            pair.second.c,
        );

        // cg_gcc as the caller, GCC as the callee.
        let from_c = c_make_big(30, 31, 32);
        if from_c.a != 30 || from_c.b != 31 || from_c.c != 32 {
            intrinsics::abort();
        }

        // GCC as the caller, cg_gcc as the callee.
        if c_call_rust() != 0 {
            intrinsics::abort();
        }

        #[cfg(target_arch = "x86_64")]
        check_c_return_register();
    }
    0
}
