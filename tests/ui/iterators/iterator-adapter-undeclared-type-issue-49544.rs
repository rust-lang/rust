//@ aux-build:iterator-adapter-undeclared-type-issue-49544.rs
//@ check-pass

extern crate iterator_adapter_undeclared_type_issue_49544 as minimal;
use minimal::foo;

fn main() {
    let _ = foo();
}

// https://github.com/rust-lang/rust/issues/49544
