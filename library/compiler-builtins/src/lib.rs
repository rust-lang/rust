#![cfg_attr(feature = "compiler-builtins", compiler_builtins)]
#![cfg_attr(not(feature = "no-asm"), feature(asm))]
#![feature(abi_unadjusted)]
#![feature(asm_experimental_arch)]
#![cfg_attr(not(feature = "no-asm"), feature(global_asm))]
#![feature(cfg_target_has_atomic)]
#![feature(compiler_builtins)]
#![feature(core_ffi_c)]
#![feature(core_intrinsics)]
#![feature(inline_const)]
#![feature(lang_items)]
#![feature(linkage)]
#![feature(naked_functions)]
#![feature(repr_simd)]
#![feature(c_unwind)]
#![cfg_attr(not(feature = "no-f16-f128"), feature(f16))]
#![cfg_attr(not(feature = "no-f16-f128"), feature(f128))]
#![no_builtins]
#![no_std]
#![allow(unused_features)]
#![allow(internal_features)]
// We use `u128` in a whole bunch of places which we currently agree with the
// compiler on ABIs and such, so we should be "good enough" for now and changes
// to the `u128` ABI will be reflected here.
#![allow(improper_ctypes, improper_ctypes_definitions)]
// `mem::swap` cannot be used because it may generate references to memcpy in unoptimized code.
#![allow(clippy::manual_swap)]
// Support compiling on both stage0 and stage1 which may differ in supported stable features.
#![allow(stable_features)]

// We disable #[no_mangle] for tests so that we can verify the test results
// against the native compiler-rt implementations of the builtins.

// NOTE cfg(all(feature = "c", ..)) indicate that compiler-rt provides an arch optimized
// implementation of that intrinsic and we'll prefer to use that

// NOTE(aapcs, aeabi, arm) ARM targets use intrinsics named __aeabi_* instead of the intrinsics
// that follow "x86 naming convention" (e.g. addsf3). Those aeabi intrinsics must adhere to the
// AAPCS calling convention (`extern "aapcs"`) because that's how LLVM will call them.

#[cfg(test)]
extern crate core;

#[macro_use]
mod macros;

pub mod float;
pub mod int;

// Disabled on x86 without sse2 due to ABI issues
// <https://github.com/rust-lang/rust/issues/114479>
#[cfg(not(all(target_arch = "x86", not(target_feature = "sse2"))))]
pub mod math;
pub mod mem;

#[cfg(target_arch = "arm")]
pub mod arm;

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
pub mod aarch64;

#[cfg(all(target_arch = "aarch64", target_os = "linux", not(feature = "no-asm"),))]
pub mod aarch64_linux;

#[cfg(all(
    kernel_user_helpers,
    any(target_os = "linux", target_os = "android"),
    target_arch = "arm"
))]
pub mod arm_linux;

#[cfg(target_arch = "hexagon")]
pub mod hexagon;

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
pub mod riscv;

#[cfg(target_arch = "x86")]
pub mod x86;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

pub mod probestack;
