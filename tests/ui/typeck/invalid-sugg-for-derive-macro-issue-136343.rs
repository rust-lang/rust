//@ proc-macro: derive-demo-issue-136343.rs

#[macro_use]
extern crate derive_demo_issue_136343;

#[derive(Sample)] //~ ERROR mismatched types
struct Test;

fn main() {}
