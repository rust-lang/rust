//@ aux-build: transient-reexports-dep.rs

#![crate_name = "bar"]

extern crate baz;

/// bar
pub use baz::Type;
