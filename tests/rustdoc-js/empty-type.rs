//@ aux-crate:emptytype=emptytype.rs
//@ compile-flags: --extern emptytype
//@ aux-build:emptytype.rs
//@ build-aux-docs

extern crate emptytype;

pub fn baz() {}
