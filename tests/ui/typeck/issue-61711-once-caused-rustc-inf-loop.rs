// Issue 61711: A crate pub re-exporting `crate` was causing an
// infinite loop.

//@ edition:2018
//@ aux-build:xcrate-issue-61711-b.rs
//@ compile-flags:--extern xcrate_issue_61711_b

//@ build-pass

fn f<F: Fn(xcrate_issue_61711_b::Struct)>(_: F) { }
fn main() { }
