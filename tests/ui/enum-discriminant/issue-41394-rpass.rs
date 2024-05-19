//@ run-pass
//@ aux-build:issue-41394.rs

extern crate issue_41394 as lib;

fn main() {
    assert_eq!(lib::foo() as u32, 42);
}
