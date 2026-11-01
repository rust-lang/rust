//! Regression test for <https://github.com/rust-lang/rust/issues/99630>.
//!
//! Re-exporting a type from another crate must not leak the private fields used
//! in the initializer of one of its associated constants.

//@ aux-build:assoc-const-private-fields.rs

#![crate_name = "foo"]

extern crate assoc_const_private_fields;

//@ has foo/struct.HasPrivateFields.html
//@ matches - '//*[@id="associatedconstant.ASSOC"]' '^pub const ASSOC: HasPrivateFields$'
pub use assoc_const_private_fields::HasPrivateFields;
