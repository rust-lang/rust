// Test that we detect changes to the `dep_kind` query. If the change is not
// detected then -Zincremental-verify-ich will trigger an assertion.

// revisions:cfail1 cfail2
// compile-flags: -Z query-dep-graph -Cpanic=unwind
// compile-pass

#![feature(panic_unwind)]

// Turn the panic_unwind crate from an explicit into an implicit query:
#[cfg(cfail1)]
extern crate panic_unwind;

fn main() {}
