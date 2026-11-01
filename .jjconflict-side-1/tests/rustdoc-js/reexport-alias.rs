//@ aux-crate:priv:reexport_alias=reexport-alias.rs
//@ compile-flags: -Zunstable-options --extern equivalent

#![crate_name = "foo"]

extern crate reexport_alias;

pub use reexport_alias::number;
