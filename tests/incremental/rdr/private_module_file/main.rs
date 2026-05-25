//@ revisions: cpass1 cpass2 cpass3
//@ compile-flags: -Z query-dep-graph -Z public-api-hash
//@ aux-build: dep.rs
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![crate_type = "bin"]
#![rustc_public_hash_unchanged(crate_name = "dep", cfg = "cpass2")]
#![rustc_public_hash_changed(crate_name = "dep", cfg = "cpass3")]

extern crate dep;

fn main() {
    dep::call_private();
}
