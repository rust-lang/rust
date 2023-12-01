#![feature(staged_api)]
#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_identity", issue = "1")]
pub const fn identity(x: i32) -> i32 {
    x
}
