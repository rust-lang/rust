#![feature(staged_api, rustc_attrs, intrinsics)]
#![stable(since="1.0.0", feature = "stable")]

#[stable(since="1.0.0", feature = "stable")]
pub mod old_way {
    extern "rust-intrinsic" {
        #[unstable(feature = "unstable", issue = "42")]
        pub fn size_of_val<T>(x: *const T) -> usize;

        #[unstable(feature = "unstable", issue = "42")]
        #[rustc_const_unstable(feature = "unstable", issue = "42")]
        pub fn min_align_of_val<T>(x: *const T) -> usize;
    }
}

#[stable(since="1.0.0", feature = "stable")]
pub mod new_way {
    #[unstable(feature = "unstable", issue = "42")]
    #[rustc_intrinsic]
    pub const unsafe fn size_of_val<T>(x: *const T) -> usize { 42 }

    #[unstable(feature = "unstable", issue = "42")]
    #[rustc_const_unstable(feature = "unstable", issue = "42")]
    #[rustc_intrinsic]
    pub const unsafe fn min_align_of_val<T>(x: *const T) -> usize { 42 }
}
