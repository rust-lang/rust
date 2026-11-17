#![feature(staged_api, rustc_attrs, intrinsics)]
#![stable(since = "1.0.0", feature = "stable")]

#[unstable(feature = "unstable", issue = "42")]
#[rustc_intrinsic]
pub const unsafe fn size_of_val<T>(x: *const T) -> usize;

#[unstable(feature = "unstable", issue = "42")]
#[rustc_intrinsic(llvm = "llvm.fshl.i64")]
pub const unsafe fn llvm_fshl(a: i64, b: i64, sh: i64) -> i64;

#[unstable(feature = "unstable", issue = "42")]
#[rustc_const_unstable(feature = "unstable", issue = "42")]
#[rustc_intrinsic]
pub const unsafe fn align_of_val<T>(x: *const T) -> usize;

#[unstable(feature = "unstable", issue = "42")]
#[rustc_const_unstable(feature = "unstable", issue = "42")]
#[rustc_intrinsic(llvm = "llvm.fshl.i64")]
pub const unsafe fn llvm_fshl2(a: i64, b: i64, sh: i64) -> i64;
