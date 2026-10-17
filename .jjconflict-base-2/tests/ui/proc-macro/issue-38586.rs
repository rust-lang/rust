//@ proc-macro: issue-38586.rs
//@ ignore-backends: gcc

#[macro_use]
extern crate issue_38586;

#[derive(A)] //~ ERROR `foo`
struct A;

fn main() {}
