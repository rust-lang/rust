//@ run-pass
//@ aux-build:generic-newtypes-cross-crate-usage-issue-9155.rs


extern crate generic_newtypes_cross_crate_usage_issue_9155 as lib;

struct Baz;

pub fn main() {
    lib::Foo::new(Baz);
}

// https://github.com/rust-lang/rust/issues/9155
