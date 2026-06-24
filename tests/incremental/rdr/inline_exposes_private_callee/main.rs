//@ revisions: cpass1 cpass2
//@ compile-flags: -Z query-dep-graph -Z public-api-hash
//@ aux-build: dep.rs
//@ ignore-backends: gcc

// A *private* function is normally excluded from the public hash (see
// `private_module_file`, where changing a private function called from a non-inline
// function does NOT change the hash). But when that same private function is reachable
// through the encoded MIR of an `#[inline]` (cross-crate-inlinable) function, it becomes
// observable by downstream crates, so changing its signature (or even just changing its span)
// will change the public hash.

#![feature(rustc_attrs)]
#![crate_type = "bin"]
#![rustc_public_hash_changed(crate_name = "dep", cfg = "cpass2")]

extern crate dep;

fn main() {
    dep::call_private();
}
