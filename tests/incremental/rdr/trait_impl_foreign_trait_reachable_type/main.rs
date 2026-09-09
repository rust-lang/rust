//@ revisions: cpass1 cpass2
//@ compile-flags: -Z query-dep-graph -Z public-api-hash
//@ aux-build: dep.rs
//@ ignore-backends: gcc

// Adding an `impl` of a *foreign* trait (here `std::fmt::Display`) for a public
// (reachable) local type must change the public hash, even though `dep` does not define
// or reexport the trait. A downstream crate can import the trait from its defining crate
// (`std`) and then call the method on `dep`'s public type, so the impl is observable.
//
// This is the positive counterpart of `trait_impl_foreign_trait_unreachable_type`, where
// the same foreign trait is implemented for a private (unreachable) type.

#![feature(rustc_attrs)]
#![crate_type = "bin"]
#![rustc_public_hash_changed(crate_name = "dep", cfg = "cpass2")]

extern crate dep;

fn main() {
    dep::anchor();
}
