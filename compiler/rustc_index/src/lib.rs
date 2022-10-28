#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
#![cfg_attr(feature = "nightly", feature(allow_internal_unstable))]
#![cfg_attr(feature = "nightly", feature(extend_one))]
#![cfg_attr(feature = "nightly", feature(min_specialization))]
#![cfg_attr(feature = "nightly", feature(new_uninit))]
#![cfg_attr(feature = "nightly", feature(step_trait))]
#![cfg_attr(feature = "nightly", feature(stmt_expr_attributes))]
#![cfg_attr(feature = "nightly", feature(test))]

#[cfg(feature = "nightly")]
pub mod bit_set;
#[cfg(feature = "nightly")]
pub mod interval;
pub mod vec;

#[cfg(feature = "rustc_macros")]
pub use rustc_macros::newtype_index;

/// Type size assertion. The first argument is a type and the second argument is its expected size.
#[macro_export]
macro_rules! static_assert_size {
    ($ty:ty, $size:expr) => {
        const _: [(); $size] = [(); ::std::mem::size_of::<$ty>()];
    };
}
