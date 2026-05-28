// Test-only use OK

//@ edition:2018
//@ check-pass
//@ aux-crate:bar=bar.rs
//@ compile-flags:--test

#![deny(unused_crate_dependencies)]

fn main() {}

#[test]
fn test_bar() {
    assert_eq!(bar::BAR, "bar");
}
