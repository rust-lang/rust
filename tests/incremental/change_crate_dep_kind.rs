// Test that we detect changes to the `dep_kind` query. If the change is not
// detected then -Zincremental-verify-ich will trigger an assertion.

//@ needs-unwind
//@ revisions: bfail1 bfail2
//@ compile-flags: -Z query-dep-graph -Cpanic=unwind
//@ needs-unwind
//@ build-pass (FIXME(62277): could be check-pass?)
//@ ignore-backends: gcc

#![cfg_attr(bfail1, feature(panic_unwind))]

// Turn the panic_unwind crate from an explicit into an implicit query:
#[cfg(bfail1)]
extern crate panic_unwind;

fn main() {}
