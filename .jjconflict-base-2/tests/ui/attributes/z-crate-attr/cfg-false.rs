// Ensure that `-Z crate-attr=cfg(false)` can comment out the whole crate
//@ compile-flags: --crate-type=lib -Zcrate-attr=cfg(false)
//@ check-pass

// NOTE: duplicate items are load-bearing
fn foo() {}
fn foo() {}
