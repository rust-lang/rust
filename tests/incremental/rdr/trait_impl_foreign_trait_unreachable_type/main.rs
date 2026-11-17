//@ revisions: cpass1 cpass2
//@ compile-flags: -Z query-dep-graph -Z public-api-hash
//@ aux-build: dep.rs
//@ ignore-backends: gcc
//@ should-fail

// Adding an `impl` of a *foreign* trait (`std::fmt::Display`) for an *unreachable*
// (private) local type should NOT change the public hash: downstream crates cannot name
// the type, so they can never observe the impl. This is the negative counterpart of
// `trait_impl_foreign_trait_reachable_type`, which implements the same foreign trait for
// a *public* (reachable) type and does change the hash.
//
// FIXME(rdr): this does not work yet. Unlike an impl of an unreachable *local* trait
// (see `trait_impl_unreachable_local_trait`, which is excluded correctly), an impl of a
// *foreign* trait is always included: a downstream crate could import the trait from its
// defining crate, so the trait itself is reachable. What is missing is gating on the
// *implementing type* being reachable. Because `Private` is not reachable, this impl can
// never apply to anything observable and should be excluded, but it currently is not. The
// assertion below is written for the *desired* behaviour (`unchanged`), which currently
// fails, so the test is marked `//@ should-fail`. Once impl inclusion is gated on the
// implementing type being reachable, this assertion will start to hold; remove
// `//@ should-fail` then.

#![feature(rustc_attrs)]
#![crate_type = "bin"]
#![rustc_public_hash_unchanged(crate_name = "dep", cfg = "cpass2")]

extern crate dep;

fn main() {
    dep::anchor();
}
