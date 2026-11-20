//! Test that multibyte Unicode characters don't crash the compiler.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/4780>.

//@ run-pass

pub fn main() {
    println!("마이너스 사인이 없으면");
}
