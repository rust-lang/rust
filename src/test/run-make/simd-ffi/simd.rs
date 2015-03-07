// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ensures that public symbols are not removed completely
#![crate_type = "lib"]
// we can compile to a variety of platforms, because we don't need
// cross-compiled standard libraries.
#![feature(no_std)]
#![no_std]

#![feature(simd, simd_ffi, link_llvm_intrinsics, lang_items)]


#[repr(C)]
#[derive(Copy)]
#[simd]
pub struct f32x4(f32, f32, f32, f32);


extern {
    #[link_name = "llvm.sqrt.v4f32"]
    fn vsqrt(x: f32x4) -> f32x4;
}

pub fn foo(x: f32x4) -> f32x4 {
    unsafe {vsqrt(x)}
}

#[repr(C)]
#[derive(Copy)]
#[simd]
pub struct i32x4(i32, i32, i32, i32);


extern {
    // _mm_sll_epi32
    #[cfg(any(target_arch = "x86",
              target_arch = "x86-64"))]
    #[link_name = "llvm.x86.sse2.psll.d"]
    fn integer(a: i32x4, b: i32x4) -> i32x4;

    // vmaxq_s32
    #[cfg(any(target_arch = "arm"))]
    #[link_name = "llvm.arm.neon.vmaxs.v4i32"]
    fn integer(a: i32x4, b: i32x4) -> i32x4;
    // vmaxq_s32
    #[cfg(any(target_arch = "aarch64"))]
    #[link_name = "llvm.aarch64.neon.maxs.v4i32"]
    fn integer(a: i32x4, b: i32x4) -> i32x4;

    // just some substitute foreign symbol, not an LLVM intrinsic; so
    // we still get type checking, but not as detailed as (ab)using
    // LLVM.
    #[cfg(not(any(target_arch = "x86",
                  target_arch = "x86-64",
                  target_arch = "arm",
                  target_arch = "aarch64")))]
    fn integer(a: i32x4, b: i32x4) -> i32x4;
}

pub fn bar(a: i32x4, b: i32x4) -> i32x4 {
    unsafe {integer(a, b)}
}

#[lang = "sized"]
pub trait Sized : PhantomFn<Self> {}

#[lang = "copy"]
pub trait Copy : PhantomFn<Self> {}

#[lang="phantom_fn"]
pub trait PhantomFn<A:?Sized,R:?Sized=()> { }
impl<A:?Sized, R:?Sized, U:?Sized> PhantomFn<A,R> for U { }

mod core {
    pub mod marker {
        pub use Copy;
    }
}
