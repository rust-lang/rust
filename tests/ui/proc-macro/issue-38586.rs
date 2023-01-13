// aux-build:issue-38586.rs

#[macro_use]
extern crate issue_38586;

#[derive(A)] //~ ERROR `foo`
struct A;

fn main() {}
