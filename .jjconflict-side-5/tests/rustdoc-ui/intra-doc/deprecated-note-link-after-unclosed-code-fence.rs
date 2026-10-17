//@ check-pass

// Regression test for https://github.com/rust-lang/rust/issues/157326.
#![allow(rustdoc::invalid_rust_codeblocks)]
#![deprecated(note = "use [Env::try_invoke] instead")]
//! ```

pub struct Env;

impl Env {
    pub fn try_invoke(&self) {}
}
