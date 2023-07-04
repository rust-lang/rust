// compile-flags: --test
// aux-build: issue-100263-macro.rs
#![feature(custom_test_frameworks)]

extern crate issue_100263_macro;

#[test_case]
#[issue_100263_macro::test]
fn foo() {}
//~^ ERROR mismatched types

fn main() {}
