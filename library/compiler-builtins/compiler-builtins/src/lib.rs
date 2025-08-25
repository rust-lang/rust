#![cfg_attr(feature = "compiler-builtins", compiler_builtins)]
#![cfg_attr(all(target_family = "wasm"), feature(wasm_numeric_instr))]
#![feature(abi_custom)]
#![feature(abi_unadjusted)]
#![feature(asm_experimental_arch)]
#![feature(cfg_target_has_atomic)]
#![feature(compiler_builtins)]
#![feature(core_intrinsics)]
#![feature(linkage)]
#![feature(asm_cfg)]
#![feature(naked_functions)]
#![feature(repr_simd)]
#![feature(macro_metavar_expr_concat)]
#![feature(rustc_attrs)]
#![cfg_attr(f16_enabled, feature(f16))]
#![cfg_attr(f128_enabled, feature(f128))]
#![no_builtins]
#![no_std]
#![allow(unused_features)]
#![allow(internal_features)]
// `mem::swap` cannot be used because it may generate references to memcpy in unoptimized code.
#![allow(clippy::manual_swap)]
// Support compiling on both stage0 and stage1 which may differ in supported stable features.
#![allow(stable_features)]
// By default, disallow this as it is forbidden in edition 2024. There is a lot of unsafe code to
// be migrated, however, so exceptions exist.
#![warn(unsafe_op_in_unsafe_fn)]

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
pub mod math;
pub mod mem;

// `libm` expects its `support` module to be available in the crate root.
use math::libm_math::support;

#[cfg(target_arch = "arm")]
pub mod arm;

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
pub mod aarch64;

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
pub mod aarch64_linux;

#[cfg(all(
    kernel_user_helpers,
    any(target_os = "linux", target_os = "android"),
    target_arch = "arm"
))]
pub mod arm_linux;

#[cfg(target_arch = "avr")]
pub mod avr;

#[cfg(target_arch = "hexagon")]
pub mod hexagon;

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
pub mod riscv;

#[cfg(target_arch = "x86")]
pub mod x86;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

pub mod probestack;
