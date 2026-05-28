#![feature(staged_api)]
#![stable(feature = "foo_lib", since = "1.0.0")]

#[stable(feature = "old_foo", since = "1.0.1")]
pub mod old {
    /// Old, stable foo
    #[stable(feature = "old_foo", since = "1.0.1")]
    pub fn foo() {}
}

#[unstable(feature = "new_foo", issue = "none")]
pub mod new {
    /// New, unstable foo
    #[unstable(feature = "new_foo", issue = "none")]
    pub fn foo() {}
}
