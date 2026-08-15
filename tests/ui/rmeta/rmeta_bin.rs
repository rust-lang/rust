//@ build-fail
//@ compile-flags: --crate-type=bin
//@ aux-build:rmeta-meta.rs
//@ no-prefer-dynamic

// Check that building a bin crate fails if a dependent crate is metadata-only.

extern crate rmeta_meta;
use rmeta_meta::Foo;

fn main() {
    let _ = Foo { field: 42 };
}

//~? ERROR crate `rmeta_meta` required to be available in rlib format, but was not found in this form
