//@ compile-flags: -l foo:bar -l foo:baz

#![crate_type = "lib"]

#[link(name = "foo")]
extern "C" {}

//~? ERROR multiple renamings were specified for library `foo`
