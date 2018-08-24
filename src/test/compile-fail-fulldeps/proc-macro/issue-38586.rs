// aux-build:issue_38586.rs
// ignore-stage1

#[macro_use]
extern crate issue_38586;

#[derive(A)] //~ ERROR `foo`
struct A;

fn main() {}
