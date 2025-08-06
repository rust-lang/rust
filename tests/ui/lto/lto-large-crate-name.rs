// This file has very long lines, but there is no way to avoid it as we are testing
// long crate names. so:
// ignore-tidy-linelength
//@ build-pass
//@ aux-build:lto-large-large-large-large-large-large-large-large-large-large-large-large-large-large-large-large-large-crate-name.rs
//@ compile-flags: -C lto
//@ no-prefer-dynamic
extern crate lto_large_large_large_large_large_large_large_large_large_large_large_large_large_large_large_large_large_crate_name;

fn main() {}
