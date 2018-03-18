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
#![feature(repr_simd)]
#![feature(abi_unadjusted)]
#![feature(linkage)]
#![feature(lang_items)]
#![allow(unused_features)]
#![no_builtins]
#![cfg_attr(feature = "compiler-builtins", feature(staged_api))]
#![cfg_attr(feature = "compiler-builtins",
            unstable(feature = "compiler_builtins_lib",
                     reason = "Compiler builtins. Will never become stable.",
                     issue = "0"))]

// We disable #[no_mangle] for tests so that we can verify the test results
// against the native compiler-rt implementations of the builtins.

// NOTE cfg(all(feature = "c", ..)) indicate that compiler-rt provides an arch optimized
// implementation of that intrinsic and we'll prefer to use that

// NOTE(aapcs, aeabi, arm) ARM targets use intrinsics named __aeabi_* instead of the intrinsics
// that follow "x86 naming convention" (e.g. addsf3). Those aeabi intrinsics must adhere to the
// AAPCS calling convention (`extern "aapcs"`) because that's how LLVM will call them.

#[cfg(test)]
extern crate core;

fn abort() -> ! {
    unsafe { core::intrinsics::abort() }
}

#[macro_use]
mod macros;

pub mod int;
pub mod float;

pub mod mem;

#[cfg(target_arch = "arm")]
pub mod arm;

#[cfg(all(kernel_user_helpers, target_os = "linux", target_arch = "arm"))]
pub mod arm_linux;

#[cfg(target_arch = "x86")]
pub mod x86;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

pub mod probestack;
