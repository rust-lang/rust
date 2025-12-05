//@ aux-build:issue-38875-b.rs
//@ check-pass

extern crate issue_38875_b;

fn main() {
    let test_x = [0; issue_38875_b::FOO];
}
