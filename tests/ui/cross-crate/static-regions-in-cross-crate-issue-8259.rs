//@ run-pass
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

//@ aux-build:static-regions-in-cross-crate-issue-8259.rs


extern crate static_regions_in_cross_crate_issue_8259 as other;
static a: other::Foo<'static> = other::Foo::A;

pub fn main() {}

// https://github.com/rust-lang/rust/issues/8259
