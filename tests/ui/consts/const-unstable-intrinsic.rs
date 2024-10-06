//! Ensure that unstable intrinsics can actually not be called,
//! neither within a crate nor cross-crate.
//@ aux-build:unstable_intrinsic.rs
#![feature(staged_api, rustc_attrs, intrinsics)]
#![stable(since="1.0.0", feature = "stable")]
#![feature(local)]

extern crate unstable_intrinsic;

fn main() {
    const_main();
}

const fn const_main() {
    let x = 42;
    unsafe {
        unstable_intrinsic::old_way::size_of_val(&x);
        //~^ERROR: unstable library feature 'unstable'
        //~|ERROR: cannot call non-const intrinsic
        unstable_intrinsic::old_way::min_align_of_val(&x);
        //~^ERROR: unstable library feature 'unstable'
        //~|ERROR: not yet stable as a const intrinsic
        unstable_intrinsic::new_way::size_of_val(&x);
        //~^ERROR: unstable library feature 'unstable'
        //~|ERROR: cannot be (indirectly) exposed to stable
        unstable_intrinsic::new_way::min_align_of_val(&x);
        //~^ERROR: unstable library feature 'unstable'
        //~|ERROR: not yet stable as a const intrinsic

        old_way::size_of_val(&x);
        //~^ERROR: cannot call non-const intrinsic
        old_way::min_align_of_val(&x);
        //~^ERROR: cannot use `#[feature(local)]`
        new_way::size_of_val(&x);
        //~^ERROR: cannot be (indirectly) exposed to stable
        new_way::min_align_of_val(&x);
        //~^ERROR: cannot use `#[feature(local)]`
    }
}

#[stable(since="1.0.0", feature = "stable")]
pub mod old_way {
    extern "rust-intrinsic" {
        #[unstable(feature = "local", issue = "42")]
        pub fn size_of_val<T>(x: *const T) -> usize;

        #[unstable(feature = "local", issue = "42")]
        #[rustc_const_unstable(feature = "local", issue = "42")]
        pub fn min_align_of_val<T>(x: *const T) -> usize;
    }
}

#[stable(since="1.0.0", feature = "stable")]
pub mod new_way {
    #[unstable(feature = "local", issue = "42")]
    #[rustc_intrinsic]
    pub const unsafe fn size_of_val<T>(x: *const T) -> usize { 42 }

    #[unstable(feature = "local", issue = "42")]
    #[rustc_const_unstable(feature = "local", issue = "42")]
    #[rustc_intrinsic]
    pub const unsafe fn min_align_of_val<T>(x: *const T) -> usize { 42 }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.63.0")]
#[inline]
pub const unsafe fn copy<T>(src: *const T, dst: *mut T, count: usize) {
    // Const stability attributes are not inherited from parent items.
    extern "rust-intrinsic" {
        fn copy<T>(src: *const T, dst: *mut T, count: usize);
    }

    unsafe { copy(src, dst, count) }
    //~^ ERROR cannot call non-const intrinsic
}
