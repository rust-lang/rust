#![feature(asm)]
#![feature(core_intrinsics)]
#![feature(linkage)]
#![feature(naked_functions)]
#![cfg_attr(not(test), no_std)]
#![no_builtins]
// TODO(rust-lang/rust#35021) uncomment when that PR lands
// #![feature(rustc_builtins)]

// We disable #[no_mangle] for tests so that we can verify the test results
// against the native compiler-rt implementations of the builtins.

// NOTE cfg(all(feature = "c", ..)) indicate that compiler-rt provides an arch optimized
// implementation of that intrinsic and we'll prefer to use that

macro_rules! udiv {
    ($a:expr, $b:expr) => {
        unsafe {
            let a = $a;
            let b = $b;

            if b == 0 {
                intrinsics::abort()
            } else {
                intrinsics::unchecked_div(a, b)
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
                intrinsics::abort()
            } else {
                intrinsics::unchecked_div(a, b)
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
                intrinsics::abort()
            } else {
                intrinsics::unchecked_rem(a, b)
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
                intrinsics::abort()
            } else {
                intrinsics::unchecked_rem(a, b)
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
