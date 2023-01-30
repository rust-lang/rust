// build-aux-docs
// ignore-cross-compile

#![crate_name = "foo"]

// @has foo/fn.foo.html '//code' ''
#[doc = "Attempted to pass a string containing `\0`"]
pub fn foo() {}
