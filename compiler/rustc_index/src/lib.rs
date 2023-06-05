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
#[macro_export]
macro_rules! static_assert_size {
    ($ty:ty, $size:expr) => {
        const _: [(); $size] = [(); ::std::mem::size_of::<$ty>()];
    };
}
