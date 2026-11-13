//! Intrinsics that cannot be moved to `core` because they depend on `alloc` types.
#![unstable(feature = "liballoc_internals", issue = "none")]

use core::mem::MaybeUninit;

use crate::boxed::Box;

/// Writes `x` into `b`.
///
/// This is needed for `vec!`, which can't afford any extra copies of the argument (or else debug
/// builds regress), has to be written fully as a call chain without `let` (or else this breaks inference
/// of e.g. unsizing coercions), and can't use an `unsafe` block as that would then also
/// include the user-provided `$x`.
#[rustc_intrinsic]
pub fn write_box_via_move<T>(b: Box<MaybeUninit<T>>, x: T) -> Box<MaybeUninit<T>>;
