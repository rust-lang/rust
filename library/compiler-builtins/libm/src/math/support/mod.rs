#[macro_use]
pub mod macros;
mod big;
mod env;
// Runtime feature detection requires atomics.
#[cfg(target_has_atomic = "ptr")]
pub(crate) mod feature_detect;
mod float_traits;
pub mod hex_float;
mod int_traits;
mod modular;

#[allow(unused_imports)]
pub use big::{i256, u256};
// Clippy seems to have a false positive
#[allow(unused_imports, clippy::single_component_path_imports)]
pub(crate) use cfg_if;
pub use env::{FpResult, Round, Status};
#[allow(unused_imports)]
pub use float_traits::{DFloat, Float, HFloat, IntTy};
pub(crate) use float_traits::{f32_from_bits, f64_from_bits};
#[cfg(any(test, feature = "unstable-public-internals"))]
pub use hex_float::Hexf;
#[cfg(f16_enabled)]
#[allow(unused_imports)]
pub use hex_float::hf16;
#[cfg(f128_enabled)]
#[allow(unused_imports)]
pub use hex_float::hf128;
#[allow(unused_imports)]
pub use hex_float::{hf32, hf64};
pub use int_traits::{CastFrom, CastInto, DInt, HInt, Int, MinInt, NarrowingDiv};
pub use modular::linear_mul_reduction;

/// Hint to the compiler that the current path is cold.
pub fn cold_path() {
    #[cfg(intrinsics_enabled)]
    core::hint::cold_path();
}

/// # Safety
///
/// `y` must not be zero and the result must not overflow (`x > i32::MIN`).
pub unsafe fn unchecked_div_i32(x: i32, y: i32) -> i32 {
    cfg_if! {
        if #[cfg(all(not(debug_assertions), intrinsics_enabled))] {
            // Temporary macro to avoid panic codegen for division (in debug mode too). At
            // the time of this writing this is only used in a few places, and once
            // rust-lang/rust#72751 is fixed then this macro will no longer be necessary and
            // the native `/` operator can be used and panics won't be codegen'd.
            //
            // Note: I am not sure whether the above comment is still up to date, we need
            // to double check whether panics are elided where we use this.
            unsafe { core::intrinsics::unchecked_div(x, y) }
        } else {
            x / y
        }
    }
}

/// # Safety
///
/// `y` must not be zero and the result must not overflow (`x > isize::MIN`).
pub unsafe fn unchecked_div_isize(x: isize, y: isize) -> isize {
    cfg_if! {
        if #[cfg(all(not(debug_assertions), intrinsics_enabled))] {
            unsafe { core::intrinsics::unchecked_div(x, y) }
        } else {
            x / y
        }
    }
}
