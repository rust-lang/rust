//! Checks that `Weak` pointers can be created with an empty enum type parameter.
//! And generic `Weak` handles zero-variant enums without error.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/48493>

//@ run-pass

enum Void {}

fn main() {
    let _ = std::rc::Weak::<Void>::new();
    let _ = std::sync::Weak::<Void>::new();
}
