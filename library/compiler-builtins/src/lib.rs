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
#![feature(linkage)]
#![feature(naked_functions)]
#![feature(staged_api)]
#![no_builtins]
#![unstable(feature = "compiler_builtins_lib",
            reason = "Compiler builtins. Will never become stable.",
            issue = "0")]

// We disable #[no_mangle] for tests so that we can verify the test results
// against the native compiler-rt implementations of the builtins.

// NOTE cfg(all(feature = "c", ..)) indicate that compiler-rt provides an arch optimized
// implementation of that intrinsic and we'll prefer to use that

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

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(test)]
extern crate core;

#[cfg(test)]
extern crate gcc_s;

#[cfg(test)]
extern crate compiler_rt;

#[cfg(test)]
extern crate rand;

#[cfg(feature = "weak")]
extern crate rlibc;

#[cfg(test)]
#[macro_use]
mod qc;

pub mod int;
pub mod float;

#[cfg(target_arch = "arm")]
pub mod arm;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;
