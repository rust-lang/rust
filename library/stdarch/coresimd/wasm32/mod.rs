//! WASM32 intrinsics

#![allow(deprecated)]

#[macro_use]
#[cfg(all(not(test), feature = "wasm_simd128"))]
mod simd128;

#[cfg(all(test, feature = "wasm_simd128"))]
pub mod simd128;
#[cfg(all(test, feature = "wasm_simd128"))]
pub use self::simd128::*;

#[cfg(test)]
use stdsimd_test::assert_instr;
#[cfg(test)]
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(any(target_feature = "atomics", dox))]
mod atomic;
#[cfg(any(target_feature = "atomics", dox))]
pub use self::atomic::*;

mod memory;
pub use self::memory::*;

/// Generates the trap instruction `UNREACHABLE`
#[cfg_attr(test, assert_instr(unreachable))]
#[inline]
pub unsafe fn unreachable() -> ! {
    ::intrinsics::abort()
}
