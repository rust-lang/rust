#![feature(intrinsics, staged_api, const_intrinsic_copy)]
#![stable(feature = "core", since = "1.6.0")]

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_intrinsic_copy", issue = "80697")]
#[inline]
pub const unsafe fn copy<T>(src: *const T, dst: *mut T, count: usize) {
    // Const stability attributes are not inherited from parent items.
    extern "rust-intrinsic" {
        fn copy<T>(src: *const T, dst: *mut T, count: usize);
    }

    unsafe { copy(src, dst, count) }
    //~^ ERROR calls in constant functions are limited to constant functions
}

fn main() {}
