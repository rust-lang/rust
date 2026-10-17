//@ aux-build:nested-dollar-crate.rs
//@ edition:2018
//@ run-pass

extern crate nested_dollar_crate;

fn main() {
    assert_eq!(nested_dollar_crate::inner!(), "In def crate!");
}
