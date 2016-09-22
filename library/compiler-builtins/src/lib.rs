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

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(test)]
extern crate core;

#[cfg(test)]
extern crate gcc_s;

#[cfg(test)]
extern crate rand;

#[cfg(feature = "weak")]
extern crate rlibc;

pub mod int;
pub mod float;

#[cfg(target_arch = "arm")]
pub mod arm;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(test)]
mod qc;

