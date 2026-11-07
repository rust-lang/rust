//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/20389
pub trait T {
    type C;
    fn dummy(&self) { }
}
