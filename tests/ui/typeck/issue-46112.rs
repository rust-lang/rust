// Issue 46112: An extern crate pub re-exporting libcore was causing
// paths rooted from `std` to be misrendered in the diagnostic output.

//@ aux-build:xcrate-issue-46112-rexport-core.rs

extern crate xcrate_issue_46112_rexport_core;
fn test(r: Result<Option<()>, &'static str>) { }
fn main() { test(Ok(())); }
//~^ ERROR mismatched types
