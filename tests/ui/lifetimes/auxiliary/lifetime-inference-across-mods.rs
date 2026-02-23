//! auxiliary crate for <https://github.com/rust-lang/rust/issues/11529>
pub struct A<'a>(pub &'a isize);
