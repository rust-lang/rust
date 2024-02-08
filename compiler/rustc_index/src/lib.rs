#![cfg_attr(
    feature = "nightly",
    feature(extend_one, min_specialization, new_uninit, step_trait, test)
)]
#![cfg_attr(all(feature = "nightly", test), feature(stmt_expr_attributes))]
#![cfg_attr(feature = "nightly", allow(internal_features))]

pub mod bit_set;
#[cfg(feature = "nightly")]
pub mod interval;

mod idx;
mod slice;
mod vec;

pub use {idx::Idx, slice::IndexSlice, vec::IndexVec};

pub use rustc_index_macros::newtype_index;

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
