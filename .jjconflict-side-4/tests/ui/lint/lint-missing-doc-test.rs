//! This test checks that denying the missing_docs lint does not trigger
//! on the generated test harness.

//@ check-pass
//@ compile-flags: --test

#![forbid(missing_docs)]

#[test]
fn test() {}
