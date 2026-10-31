//@ aux-crate:zzzzz=zzzzz.rs
//@ aux-build:zzzzz.rs
//@ build-aux-docs
//@ edition:2021

#![crate_name = "foo"]

#[doc(no_inline)]
pub use zzzzz::Eepy;
