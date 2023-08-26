#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
#![cfg_attr(
    feature = "nightly",
    feature(
        allow_internal_unstable,
        extend_one,
        min_specialization,
        new_uninit,
        step_trait,
        stmt_expr_attributes,
        test
    )
)]
#![cfg_attr(feature = "nightly", allow(internal_features))]

#[cfg(feature = "nightly")]
pub mod bit_set;
#[cfg(feature = "nightly")]
pub mod interval;

mod idx;
mod slice;
mod vec;

pub use {idx::Idx, slice::IndexSlice, vec::IndexVec};

#[cfg(feature = "rustc_macros")]
pub use rustc_macros::newtype_index;

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
macro_rules! static_assert_size {
    ($ty:ty, $size:expr) => {
        const _: [(); $size] = [(); ::std::mem::size_of::<$ty>()];
    };
}
