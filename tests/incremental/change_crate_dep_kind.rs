// Test that we detect changes to the `dep_kind` query. If the change is not
// detected then -Zincremental-verify-ich will trigger an assertion.

//@ needs-unwind
//@ revisions: bpass1 bpass2
//@ compile-flags: -Z query-dep-graph -Cpanic=unwind
//@ needs-unwind
//@ ignore-backends: gcc
// FIXME(#62277): could be check-pass?

#![cfg_attr(bpass1, feature(panic_unwind))]

// Turn the panic_unwind crate from an explicit into an implicit query:
#[cfg(bpass1)]
extern crate panic_unwind;

fn main() {}
