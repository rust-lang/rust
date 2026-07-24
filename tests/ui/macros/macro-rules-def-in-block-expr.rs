//! Regression test for <https://github.com/rust-lang/rust/issues/39709>.
//! Macro definition inside block expression caused ICE.
//@ run-pass

#![allow(unused_macros)]
fn main() {
    println!("{}", { macro_rules! x { ($(t:tt)*) => {} } 33 });
}
