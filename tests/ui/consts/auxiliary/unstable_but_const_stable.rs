#![feature(staged_api, rustc_attrs, intrinsics)]
#![stable(since="1.0.0", feature = "stable")]

extern "rust-intrinsic" {
    #[unstable(feature = "unstable", issue = "42")]
    #[rustc_const_stable(feature = "stable", since = "1.0.0")]
    #[rustc_nounwind]
    pub fn write_bytes<T>(dst: *mut T, val: u8, count: usize);
}

#[unstable(feature = "unstable", issue = "42")]
#[rustc_const_stable(feature = "stable", since = "1.0.0")]
pub const fn some_unstable_fn() {}
