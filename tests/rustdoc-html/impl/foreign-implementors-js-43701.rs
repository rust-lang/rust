// https://github.com/rust-lang/rust/issues/43701
#![crate_name = "foo"]

pub use std::vec::Vec;

//@ !has trait.impl/core/clone/trait.Clone.js
