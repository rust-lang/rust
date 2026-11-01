//! Regression test for a crash caused by an "unsused move"
//! (specifically, a variable bound to a `Box` used as a statement)
//! leading to incorrect memory zero-filling after drop.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/3878>.

//@ run-pass

pub fn main() {
    let y: Box<_> = Box::new(1);
    drop(y);
}
