// check-pass

#![feature(intrinsics, staged_api, const_intrinsic_copy)]
#![stable(feature = "core", since = "1.6.0")]

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_intrinsic_copy", issue = "80697")]
#[inline]
pub const unsafe fn copy<T>(src: *const T, dst: *mut T, count: usize) {
    extern "rust-intrinsic" {
        fn copy<T>(src: *const T, dst: *mut T, count: usize);
    }

    // Even though the `copy` intrinsic lacks stability attributes, this works, because it
    // inherits its stability attributes from its parent. That includes `rustc_const_(un)stable`
    // attributes.
    unsafe { copy(src, dst, count) }
}

fn main() {}
