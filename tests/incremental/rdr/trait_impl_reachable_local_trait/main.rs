//@ revisions: cpass1 cpass2
//@ compile-flags: -Z query-dep-graph -Z public-api-hash
//@ aux-build: dep.rs
//@ ignore-backends: gcc

// Adding an `impl` of a *reachable* (public) local trait for a public type adds a new
// observable piece of API: downstream crates can import the trait and call the method
// on the type. So the public hash MUST change.
//
// This is the positive counterpart of `trait_impl_unreachable_local_trait`, where the
// trait is private (unreachable) and the impl should therefore be invisible.

#![feature(rustc_attrs)]
#![crate_type = "bin"]
#![rustc_public_hash_changed(crate_name = "dep", cfg = "cpass2")]

extern crate dep;

fn main() {
    dep::anchor();
}
