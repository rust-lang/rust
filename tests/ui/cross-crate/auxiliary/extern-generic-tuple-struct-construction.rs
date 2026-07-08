//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/4545
pub struct S<T>(Option<T>);
pub fn mk<T>() -> S<T> { S(None) }
