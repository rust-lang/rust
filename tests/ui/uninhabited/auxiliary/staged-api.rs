#![feature(staged_api)]
#![stable(feature = "stable", since = "1.0.0")]

#[stable(feature = "stable", since = "1.0.0")]
pub struct Foo<T> {
    #[unstable(feature = "unstable", issue = "none")]
    pub field: T,
}
