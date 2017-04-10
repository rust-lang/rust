#![cfg_attr(not(stage0), deny(warnings))]
#![cfg_attr(not(test), no_std)]
#![cfg_attr(feature = "compiler-builtins", compiler_builtins)]
#![crate_name = "compiler_builtins"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/",
       test(attr(deny(warnings))))]
#![feature(asm)]
#![feature(compiler_builtins)]
#![feature(core_intrinsics)]
#![feature(naked_functions)]
#![feature(staged_api)]
#![feature(i128_type)]
#![feature(repr_simd)]
#![feature(abi_unadjusted)]
#![allow(unused_features)]
#![no_builtins]
#![unstable(feature = "compiler_builtins_lib",
            reason = "Compiler builtins. Will never become stable.",
            issue = "0")]

// We disable #[no_mangle] for tests so that we can verify the test results
// against the native compiler-rt implementations of the builtins.

// NOTE cfg(all(feature = "c", ..)) indicate that compiler-rt provides an arch optimized
// implementation of that intrinsic and we'll prefer to use that

// NOTE(aapcs, aeabi, arm) ARM targets use intrinsics named __aeabi_* instead of the intrinsics
// that follow "x86 naming convention" (e.g. addsf3). Those aeabi intrinsics must adhere to the
// AAPCS calling convention (`extern "aapcs"`) because that's how LLVM will call them.

// TODO(rust-lang/rust#37029) use e.g. checked_div(_).unwrap_or_else(|| abort())
macro_rules! udiv {
    ($a:expr, $b:expr) => {
        unsafe {
            let a = $a;
            let b = $b;

            if b == 0 {
                ::core::intrinsics::abort()
            } else {
                ::core::intrinsics::unchecked_div(a, b)
            }
        }
    }
}

macro_rules! sdiv {
    ($sty:ident, $a:expr, $b:expr) => {
        unsafe {
            let a = $a;
            let b = $b;

            if b == 0 || (b == -1 && a == $sty::min_value()) {
                ::core::intrinsics::abort()
            } else {
                ::core::intrinsics::unchecked_div(a, b)
            }
        }
    }
}

macro_rules! urem {
    ($a:expr, $b:expr) => {
        unsafe {
            let a = $a;
            let b = $b;

            if b == 0 {
                ::core::intrinsics::abort()
            } else {
                ::core::intrinsics::unchecked_rem(a, b)
            }
        }
    }
}

macro_rules! srem {
    ($sty:ty, $a:expr, $b:expr) => {
        unsafe {
            let a = $a;
            let b = $b;

            if b == 0 || (b == -1 && a == $sty::min_value()) {
                ::core::intrinsics::abort()
            } else {
                ::core::intrinsics::unchecked_rem(a, b)
            }
        }
    }
}

// Hack for LLVM expectations for ABI on windows
#[cfg(all(windows, target_pointer_width="64"))]
#[repr(simd)]
pub struct U64x2(u64, u64);

#[cfg(all(windows, target_pointer_width="64"))]
fn conv(i: u128) -> U64x2 {
    use int::LargeInt;
    U64x2(i.low(), i.high())
}

#[cfg(all(windows, target_pointer_width="64"))]
fn sconv(i: i128) -> U64x2 {
    use int::LargeInt;
    let j = i as u128;
    U64x2(j.low(), j.high())
}

#[cfg(test)]
extern crate core;

pub mod int;
pub mod float;

#[cfg(feature = "mem")]
pub mod mem;

#[cfg(target_arch = "arm")]
pub mod arm;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;
