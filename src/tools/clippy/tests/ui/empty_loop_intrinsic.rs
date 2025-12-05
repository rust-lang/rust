//@check-pass

#![warn(clippy::empty_loop)]
#![feature(intrinsics)]
#![feature(rustc_attrs)]

// From issue #15200
#[rustc_intrinsic]
#[rustc_nounwind]
/// # Safety
pub const unsafe fn simd_insert<T, U>(x: T, idx: u32, val: U) -> T;

fn main() {}
