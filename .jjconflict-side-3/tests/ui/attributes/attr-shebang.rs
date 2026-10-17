//! Check that we accept crate-level inner attributes with the `#![..]` shebang syntax.

//@ check-pass

#![allow(stable_features)]
#![feature(rust1)]
pub fn main() { }
