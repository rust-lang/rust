//@ run-pass
//@ aux-build:issue-12133-rlib.rs
//@ aux-build:issue-12133-dylib.rs
//@ aux-build:issue-12133-dylib2.rs
//@ needs-crate-type: dylib
//@ needs-dynamic-linking


extern crate issue_12133_dylib2 as other;

fn main() {}
