//! Test that basic multiline comments are parsed correctly.
//!
//! Feature implementation test for <https://github.com/rust-lang/rust/issues/66>.

//@ run-pass

/*
 * This is a multi-line comment.
 */
pub fn main() {}
