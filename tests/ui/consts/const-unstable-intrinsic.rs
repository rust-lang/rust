//! Ensure that unstable intrinsics can actually not be called,
//! neither within a crate nor cross-crate.
//@ aux-build:unstable_intrinsic.rs
#![feature(staged_api, rustc_attrs, intrinsics)]
#![stable(since = "1.0.0", feature = "stable")]
#![feature(local)]

extern crate unstable_intrinsic;

fn main() {
    const_main();
}

const fn const_main() {
    let x = 42;
    unsafe {
        unstable_intrinsic::size_of_val(&x);
        //~^ERROR: unstable library feature `unstable`
        //~|ERROR: not yet stable as a const intrinsic
        unstable_intrinsic::align_of_val(&x);
        //~^ERROR: unstable library feature `unstable`
        //~|ERROR: not yet stable as a const intrinsic

        size_of_val(&x);
        //~^ERROR: cannot use `#[feature(local)]`
        align_of_val(&x);
        //~^ERROR: cannot use `#[feature(local)]`
    }
}

#[unstable(feature = "local", issue = "42")]
#[rustc_intrinsic]
pub const unsafe fn size_of_val<T>(x: *const T) -> usize;

#[unstable(feature = "local", issue = "42")]
#[rustc_const_unstable(feature = "local", issue = "42")]
#[rustc_intrinsic]
pub const unsafe fn align_of_val<T>(x: *const T) -> usize;

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.63.0")]
#[inline]
pub const unsafe fn copy<T>(src: *const T, dst: *mut T, count: usize) {
    // Const stability attributes are not inherited from parent items.
    #[rustc_intrinsic]
    const unsafe fn copy<T>(src: *const T, dst: *mut T, count: usize);

    unsafe { copy(src, dst, count) }
    //~^ ERROR cannot be (indirectly) exposed to stable
}

// Ensure that a fallback body is recursively-const-checked.
mod fallback {
    #[rustc_intrinsic]
    const unsafe fn copy<T>(src: *const T, _dst: *mut T, _count: usize) {
        super::size_of_val(src);
        //~^ ERROR cannot use `#[feature(local)]`
    }
}
