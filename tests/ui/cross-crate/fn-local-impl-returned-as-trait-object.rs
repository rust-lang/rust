//@ run-pass
//@ aux-build:fn-local-impl-returned-as-trait-object.rs
//! Regression test for https://github.com/rust-lang/rust/issues/2380
//! This test exposes a bug where a function that defines a trait impl inside its body
//!  and returns it as a trait object failed  because the reachability pass skipped nested
//! items in function bodies.

extern crate a;

pub fn main() {
    a::f::<()>();
}
