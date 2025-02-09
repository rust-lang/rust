#[cfg(test)]
use stdarch_test::assert_instr;

unsafe extern "C" {
    #[link_name = "llvm.wasm.memory.grow"]
    fn llvm_memory_grow(mem: u32, pages: usize) -> usize;
    #[link_name = "llvm.wasm.memory.size"]
    fn llvm_memory_size(mem: u32) -> usize;
}

/// Corresponding intrinsic to wasm's [`memory.size` instruction][instr]
///
/// This function, when called, will return the current memory size in units of
/// pages. The current WebAssembly page size is 65536 bytes (64 KB).
///
/// The argument `MEM` is the numerical index of which memory to return the
/// size of. Note that currently the WebAssembly specification only supports one
/// memory, so it is required that zero is passed in. The argument is present to
/// be forward-compatible with future WebAssembly revisions. If a nonzero
/// argument is passed to this function it will currently unconditionally abort.
///
/// [instr]: http://webassembly.github.io/spec/core/exec/instructions.html#exec-memory-size
#[inline]
#[cfg_attr(test, assert_instr("memory.size", MEM = 0))]
#[rustc_legacy_const_generics(0)]
#[stable(feature = "simd_wasm32", since = "1.33.0")]
#[doc(alias("memory.size"))]
pub fn memory_size<const MEM: u32>() -> usize {
    static_assert!(MEM == 0);
    unsafe { llvm_memory_size(MEM) }
}

/// Corresponding intrinsic to wasm's [`memory.grow` instruction][instr]
///
/// This function, when called, will attempt to grow the default linear memory
/// by the specified `delta` of pages. The current WebAssembly page size is
/// 65536 bytes (64 KB). If memory is successfully grown then the previous size
/// of memory, in pages, is returned. If memory cannot be grown then
/// `usize::MAX` is returned.
///
/// The argument `MEM` is the numerical index of which memory to return the
/// size of. Note that currently the WebAssembly specification only supports one
/// memory, so it is required that zero is passed in. The argument is present to
/// be forward-compatible with future WebAssembly revisions. If a nonzero
/// argument is passed to this function it will currently unconditionally abort.
///
/// [instr]: http://webassembly.github.io/spec/core/exec/instructions.html#exec-memory-grow
#[inline]
#[cfg_attr(test, assert_instr("memory.grow", MEM = 0))]
#[rustc_legacy_const_generics(0)]
#[stable(feature = "simd_wasm32", since = "1.33.0")]
#[doc(alias("memory.grow"))]
pub fn memory_grow<const MEM: u32>(delta: usize) -> usize {
    unsafe {
        static_assert!(MEM == 0);
        llvm_memory_grow(MEM, delta)
    }
}
