#![feature(staged_api, rustc_attrs, intrinsics)]
#![stable(since="1.0.0", feature = "stable")]

#[unstable(feature = "unstable", issue = "42")]
#[rustc_intrinsic]
pub const unsafe fn size_of_val<T>(x: *const T) -> usize;

#[unstable(feature = "unstable", issue = "42")]
#[rustc_const_unstable(feature = "unstable", issue = "42")]
#[rustc_intrinsic]
pub const unsafe fn align_of_val<T>(x: *const T) -> usize;
