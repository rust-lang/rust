//! WASM32 intrinsics

#[cfg(test)]
use stdarch_test::assert_instr;

mod atomic;
pub use self::atomic::*;

mod simd128;
pub use self::simd128::*;

mod memory;
pub use self::memory::*;

/// Generates the trap instruction `UNREACHABLE`
#[cfg_attr(test, assert_instr(unreachable))]
#[inline]
#[stable(feature = "unreachable_wasm32", since = "1.37.0")]
pub unsafe fn unreachable() -> ! {
    crate::intrinsics::abort()
}
