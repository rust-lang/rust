//@ aux-build: transitive_dep.rs
//@ compile-flags: -Z public-api-hash

#![crate_name = "dep"]
#![crate_type = "rlib"]

extern crate transitive_dep;

pub use transitive_dep::print;
