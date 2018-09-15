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

#[inline]
#[cfg_attr(test, assert_instr("memory.size"))]
#[rustc_deprecated(reason = "renamed to memory::size", since = "1.30.0")]
#[unstable(feature = "stdsimd", issue = "27731")]
#[allow(deprecated)]
#[doc(hidden)]
pub unsafe fn current_memory() -> i32 {
    memory::size(0)
}

#[inline]
#[cfg_attr(test, assert_instr("memory.grow"))]
#[rustc_deprecated(reason = "renamed to memory::grow", since = "1.30.0")]
#[unstable(feature = "stdsimd", issue = "27731")]
#[allow(deprecated)]
#[doc(hidden)]
pub unsafe fn grow_memory(delta: i32) -> i32 {
    memory::grow(0, delta)
}

pub mod atomic;
pub mod memory;

/// Generates the trap instruction `UNREACHABLE`
#[cfg_attr(test, assert_instr(unreachable))]
#[inline]
pub unsafe fn unreachable() -> ! {
    ::_core::intrinsics::abort()
}
