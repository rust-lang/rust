// tidy-alphabetical-start
#![cfg_attr(all(feature = "nightly", test), feature(stmt_expr_attributes))]
#![cfg_attr(bootstrap, feature(new_zeroed_alloc))]
#![cfg_attr(feature = "nightly", allow(internal_features))]
#![cfg_attr(feature = "nightly", feature(extend_one, step_trait, test))]
#![cfg_attr(feature = "nightly", feature(new_range_api))]
// tidy-alphabetical-end

pub mod bit_set;
#[cfg(feature = "nightly")]
pub mod interval;

mod idx;
mod slice;
mod vec;

pub use idx::{Idx, IntoSliceIdx};
pub use rustc_index_macros::newtype_index;
pub use slice::IndexSlice;
#[doc(no_inline)]
pub use vec::IndexVec;

/// Type size assertion. The first argument is a type and the second argument is its expected size.
///
/// <div class="warning">
///
/// Emitting hard errors from size assertions like this is generally not
/// recommended, especially in libraries, because they can cause build failures if the layout
/// algorithm or dependencies change. Here in rustc we control the toolchain and layout algorithm,
/// so the former is not a problem. For the latter we have a lockfile as rustc is an application and
/// precompiled library.
///
/// Short version: Don't copy this macro into your own code. Use a `#[test]` instead.
///
/// </div>
#[macro_export]
#[cfg(not(feature = "rustc_randomized_layouts"))]
macro_rules! static_assert_size {
    ($ty:ty, $size:expr) => {
        const _: [(); $size] = [(); ::std::mem::size_of::<$ty>()];
    };
}

#[macro_export]
#[cfg(feature = "rustc_randomized_layouts")]
macro_rules! static_assert_size {
    ($ty:ty, $size:expr) => {
        // no effect other than using the statements.
        // struct sizes are not deterministic under randomized layouts
        const _: (usize, usize) = ($size, ::std::mem::size_of::<$ty>());
    };
}
