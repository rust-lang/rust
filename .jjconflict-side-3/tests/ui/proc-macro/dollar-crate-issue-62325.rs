//@ check-pass
//@ edition:2018
//@ compile-flags: -Z span-debug
//@ proc-macro: test-macros.rs
//@ aux-build:dollar-crate-external.rs


#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;
extern crate dollar_crate_external;

type S = u8;

macro_rules! m { () => {
    #[print_attr]
    struct A(identity!($crate::S));
}}

m!();

dollar_crate_external::issue_62325!();

fn main() {}
