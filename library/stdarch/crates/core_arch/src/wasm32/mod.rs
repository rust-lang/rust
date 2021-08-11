//! WASM32 intrinsics

#[cfg(test)]
use stdarch_test::assert_instr;

mod atomic;
pub use self::atomic::*;

mod simd128;
pub use self::simd128::*;

mod memory;
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
