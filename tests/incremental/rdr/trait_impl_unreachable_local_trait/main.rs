//@ revisions: cpass1 cpass2
//@ compile-flags: -Z query-dep-graph -Z public-api-hash
//@ aux-build: dep.rs
//@ ignore-backends: gcc

// Adding an `impl` of an *unreachable* (private) local trait should NOT change the
// public hash: downstream crates cannot name the trait, so they can never observe or
// call the impl. This is the negative counterpart of
// `trait_impl_reachable_local_trait`.

#![feature(rustc_attrs)]
#![crate_type = "bin"]
#![rustc_public_hash_unchanged(crate_name = "dep", cfg = "cpass2")]

extern crate dep;

fn main() {
    dep::anchor();
}
