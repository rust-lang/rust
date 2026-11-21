//@ compile-flags: -Znext-solver
//@ build-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

// `simd_cast_ptr` should accept vectors of thin pointers to unsafe binders.

#![feature(unsafe_binders, repr_simd, core_intrinsics)]
#![allow(incomplete_features)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::simd_cast_ptr;

fn main() {
    let x = 1u8;
    let ptrs: Simd<*const u8, 2> = Simd([&x as *const u8, std::ptr::null()]);
    let cast: Simd<*const unsafe<> u8, 2> = unsafe { simd_cast_ptr(ptrs) };
    assert!(cast.into_array()[0] as *const u8 == &x as *const u8);
}
