#![feature(staged_api, rustc_attrs, intrinsics)]
#![stable(since="1.0.0", feature = "stable")]

#[unstable(feature = "unstable", issue = "42")]
#[rustc_intrinsic]
#[rustc_intrinsic_must_be_overridden]
pub const unsafe fn size_of_val<T>(x: *const T) -> usize { 42 }

#[unstable(feature = "unstable", issue = "42")]
#[rustc_const_unstable(feature = "unstable", issue = "42")]
#[rustc_intrinsic]
#[rustc_intrinsic_must_be_overridden]
pub const unsafe fn min_align_of_val<T>(x: *const T) -> usize { 42 }
