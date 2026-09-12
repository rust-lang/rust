//@ compile-flags: -Znext-solver=globally -Znormalize-docs

#![crate_name = "foo"]

#[doc(inline)]
pub use std as other;
