//! "compile-fail/svh-uta-trait.rs" is checking that we detect a
//! change from `use foo::TraitB` to use `foo::TraitB` in the hash
//! (SVH) computation (#14132), since that will affect method
//! resolution.
//!
//! This is the downstream crate.

#![crate_name = "utb"]

extern crate uta;

pub fn foo() { assert_eq!(uta::foo::<()>(0), 3); }
