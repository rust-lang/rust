//! After the use of pattern types inside `NonNull`,
//! transmuting between a niche optimized enum wrapping a
//! generic `NonNull` and raw pointers stopped working.
//@ check-pass

use std::ptr::NonNull;
pub const fn is_null<'a, T: ?Sized>(ptr: *const T) -> bool {
    unsafe { matches!(core::mem::transmute::<*const T, Option<NonNull<T>>>(ptr), None) }
}

fn main() {}
