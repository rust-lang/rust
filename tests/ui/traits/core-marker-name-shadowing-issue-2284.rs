//@ run-pass
#![allow(dead_code)]

//! Tests that user-defined trait is prioritized in compile time over
//! the core::marker trait with the same name, allowing shadowing core traits.
//!
//! # Context
//! Original issue: https://github.com/rust-lang/rust/issues/2284
//! Original fix pull request: https://github.com/rust-lang/rust/pull/3792


trait Send {
    fn f(&self);
}

fn f<T:Send>(t: T) {
    t.f();
}

pub fn main() {
}
