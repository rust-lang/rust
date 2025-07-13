//@ run-pass
#![allow(dead_code)]
//@ aux-build:cross-crate-map-usage-issue-5521.rs



extern crate cross_crate_map_usage_issue_5521 as foo;

fn bar(a: foo::map) {
    if false {
        panic!();
    } else {
        let _b = &(*a)[&2];
    }
}

fn main() {}

// https://github.com/rust-lang/rust/issues/5521
