//@ compile-flags: -l foo:

#![crate_type = "lib"]

#[link(name = "foo")]
extern "C" {}

//~? ERROR an empty renaming target was specified for library `foo`
