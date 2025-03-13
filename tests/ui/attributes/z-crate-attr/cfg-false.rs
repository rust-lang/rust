// Ensure that `-Z crate-attr=cfg(FALSE)` can comment out the whole crate
//@ compile-flags: --crate-type=lib -Zcrate-attr=cfg(FALSE)
//@ check-pass

// NOTE: duplicate items are load-bearing
fn foo() {}
fn foo() {}
