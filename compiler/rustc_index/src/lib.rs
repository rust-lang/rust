#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
#![feature(allow_internal_unstable)]
#![feature(bench_black_box)]
#![feature(extend_one)]
#![feature(let_else)]
#![feature(min_specialization)]
#![feature(new_uninit)]
#![feature(step_trait)]
#![feature(stmt_expr_attributes)]
#![feature(test)]

pub mod bit_set;
pub mod interval;
pub mod vec;

pub use rustc_macros::newtype_index;

/// Type size assertion. The first argument is a type and the second argument is its expected size.
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
