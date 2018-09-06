//! WASM32 intrinsics


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

extern "C" {
    #[link_name = "llvm.wasm.grow.memory.i32"]
    fn llvm_grow_memory(pages: i32) -> i32;
    #[link_name = "llvm.wasm.current.memory.i32"]
    fn llvm_current_memory() -> i32;
}

/// Corresponding intrinsic to wasm's [`current_memory` instruction][instr]
///
/// This function, when called, will return the current memory size in units of
/// pages.
///
/// [instr]: https://github.com/WebAssembly/design/blob/master/Semantics.md#resizing
#[inline]
#[cfg_attr(test, assert_instr("memory.size"))]
pub unsafe fn current_memory() -> i32 {
    llvm_current_memory()
}

/// Corresponding intrinsic to wasm's [`grow_memory` instruction][instr]
///
/// This function, when called, will attempt to grow the default linear memory
/// by the specified number of pages. If memory is successfully grown then the
/// previous size of memory, in pages, is returned. If memory cannot be grown
/// then -1 is returned.
///
/// [instr]: https://github.com/WebAssembly/design/blob/master/Semantics.md#resizing
#[inline]
#[cfg_attr(test, assert_instr("memory.grow"))]
pub unsafe fn grow_memory(delta: i32) -> i32 {
    llvm_grow_memory(delta)
}

pub mod atomic;
