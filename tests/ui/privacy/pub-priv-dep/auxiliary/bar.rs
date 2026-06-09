//@ aux-crate:priv:foo=foo.rs
//@ compile-flags: -Zunstable-options

#![crate_type = "rlib"]
extern crate foo;
pub struct Bar(pub i32);
