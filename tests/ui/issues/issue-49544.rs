//@ aux-build:issue-49544.rs
//@ check-pass

extern crate issue_49544;
use issue_49544::foo;

fn main() {
    let _ = foo();
}
