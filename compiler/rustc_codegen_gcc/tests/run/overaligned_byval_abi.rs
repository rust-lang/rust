// Compiler:
//
// Run-time:
//   status: 0

// Checks that cg_gcc passes an over-aligned by-value ("byval") argument where the platform ABI
// says it goes, by calling in both directions with `tests/c/overaligned_byval_abi.c`, which is
// compiled by the real GCC.
//
// `tests/run/overaligned_byval_arg.rs` covers the Rust-visible half of the same bug. It cannot
// cover this one: with cg_gcc on both sides of a call, caller and callee place the argument at
// the same wrong offset and agree with each other.
//
// Two over-aligned arguments are used rather than one so that the failure is deterministic. A
// backend that drops `align(64)` packs the arguments at offsets 0, 24, 88 and 112 of the argument
// area; 112 - 24 = 88 is not a multiple of 64, so the two of them cannot both land on a 64-byte
// boundary however the argument area itself is aligned. With a single over-aligned argument the
// frame often happens to be 64-aligned and the bug hides.
//
// Only the values received are checked, never the address an argument landed at: which alignment
// a target gives a by-value stack slot differs between targets, but the two sides of a call
// agreeing on it does not. `overaligned_byval_arg.rs` is where the alignment itself is asserted.

#![feature(no_core)]
#![no_std]
#![no_core]
#![no_main]

extern crate mini_core;
use mini_core::*;

#[repr(C)]
struct Big {
    a: i64,
    b: i64,
    c: i64,
}

#[repr(C, align(64))]
struct Aligned {
    x: i32,
}

extern "C" {
    fn c_take_both(first: Big, second: Aligned, third: Big, fourth: Aligned) -> i32;
    fn c_call_rust() -> i32;
}

// The callee for the GCC-built caller in `c_call_rust`.
//
// `#[no_mangle]` is not only about the symbol name: it makes the symbol externally visible, which
// pins the calling convention. Without it the function has internal linkage and GCC is free to
// clone it with a changed convention at `-O3` (the symbol comes out as `...constprop.0.isra.0`),
// so the arguments never travel through the stack slots and the release build passes spuriously.
#[no_mangle]
extern "C" fn rust_take_both(first: Big, second: Aligned, third: Big, fourth: Aligned) -> i32 {
    if first.a as i32 != 1 || first.b as i32 != 2 || first.c as i32 != 3 {
        return 5;
    }
    if second.x != 42 {
        return 6;
    }
    if third.a as i32 != 4 || third.b as i32 != 5 || third.c as i32 != 6 {
        return 7;
    }
    if fourth.x != 43 {
        return 8;
    }
    0
}

#[no_mangle]
extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    // cg_gcc as the caller, GCC as the callee.
    let result = unsafe {
        c_take_both(
            Big { a: 1, b: 2, c: 3 },
            Aligned { x: 42 },
            Big { a: 4, b: 5, c: 6 },
            Aligned { x: 43 },
        )
    };
    if result != 0 {
        return result;
    }

    // GCC as the caller, cg_gcc as the callee.
    unsafe { c_call_rust() }
}
