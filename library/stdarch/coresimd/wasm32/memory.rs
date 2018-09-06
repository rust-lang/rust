#[cfg(test)]
use stdsimd_test::assert_instr;
#[cfg(test)]
use wasm_bindgen_test::wasm_bindgen_test;

extern "C" {
    #[link_name = "llvm.wasm.memory.grow.i32"]
    fn llvm_memory_grow(mem: i32, pages: i32) -> i32;
    #[link_name = "llvm.wasm.memory.size.i32"]
    fn llvm_memory_size(mem: i32) -> i32;
}

/// Corresponding intrinsic to wasm's [`memory.size` instruction][instr]
///
/// This function, when called, will return the current memory size in units of
/// pages.
///
/// The argument `mem` is the numerical index of which memory to return the
/// size of. Note that currently wasm only supports one memory, so specifying
/// a nonzero value will likely result in a runtime validation error of the
/// wasm module.
///
/// [instr]: https://github.com/WebAssembly/design/blob/master/Semantics.md#resizing
#[inline]
#[cfg_attr(test, assert_instr("memory.size", mem = 0))]
#[rustc_args_required_const(0)]
pub unsafe fn size(mem: i32) -> i32 {
    if mem != 0 {
        ::intrinsics::abort();
    }
    llvm_memory_size(0)
}

/// Corresponding intrinsic to wasm's [`memory.grow` instruction][instr]
///
/// This function, when called, will attempt to grow the default linear memory
/// by the specified `delta` of pages. If memory is successfully grown then the
/// previous size of memory, in pages, is returned. If memory cannot be grown
/// then -1 is returned.
///
/// The argument `mem` is the numerical index of which memory to return the
/// size of. Note that currently wasm only supports one memory, so specifying
/// a nonzero value will likely result in a runtime validation error of the
/// wasm module.
///
/// [instr]: https://github.com/WebAssembly/design/blob/master/Semantics.md#resizing
#[inline]
#[cfg_attr(test, assert_instr("memory.grow", mem = 0))]
#[rustc_args_required_const(0)]
pub unsafe fn grow(mem: i32, delta: i32) -> i32 {
    if mem != 0 {
        ::intrinsics::abort();
    }
    llvm_memory_grow(0, delta)
}
