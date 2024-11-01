//! WASM32 intrinsics

#[cfg(test)]
use stdarch_test::assert_instr;

mod atomic;
#[unstable(feature = "stdarch_wasm_atomic_wait", issue = "77839")]
pub use self::atomic::*;

mod simd128;
#[stable(feature = "wasm_simd", since = "1.54.0")]
pub use self::simd128::*;

mod relaxed_simd;
#[stable(feature = "stdarch_wasm_relaxed_simd", since = "1.82.0")]
pub use self::relaxed_simd::*;

mod memory;
#[stable(feature = "simd_wasm32", since = "1.33.0")]
pub use self::memory::*;

/// Generates the [`unreachable`] instruction, which causes an unconditional [trap].
///
/// This function is safe to call and immediately aborts the execution.
///
/// [`unreachable`]: https://webassembly.github.io/spec/core/syntax/instructions.html#syntax-instr-control
/// [trap]: https://webassembly.github.io/spec/core/intro/overview.html#trap
#[cfg_attr(test, assert_instr(unreachable))]
#[inline]
#[stable(feature = "unreachable_wasm32", since = "1.37.0")]
pub fn unreachable() -> ! {
    crate::intrinsics::abort()
}

extern "C-unwind" {
    #[link_name = "llvm.wasm.throw"]
    fn wasm_throw(tag: i32, ptr: *mut u8) -> !;
}

/// Generates the [`throw`] instruction from the [exception-handling proposal] for WASM.
///
/// This function is unlikely to be stabilized until codegen backends have better support.
///
/// [`throw`]: https://webassembly.github.io/exception-handling/core/syntax/instructions.html#syntax-instr-control
/// [exception-handling proposal]: https://github.com/WebAssembly/exception-handling
// FIXME: wasmtime does not currently support exception-handling, so cannot execute
//        a wasm module with the throw instruction in it. once it does, we can
//        reenable this attribute.
// #[cfg_attr(test, assert_instr(throw, TAG = 0, ptr = core::ptr::null_mut()))]
#[inline]
#[unstable(feature = "wasm_exception_handling_intrinsics", issue = "122465")]
pub unsafe fn throw<const TAG: i32>(ptr: *mut u8) -> ! {
    static_assert!(TAG == 0); // LLVM only supports tag 0 == C++ right now.
    wasm_throw(TAG, ptr)
}
