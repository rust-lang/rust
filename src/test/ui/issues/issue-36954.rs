// run-pass
// aux-build:issue-36954.rs

extern crate issue_36954 as lib;

fn main() {
    let _ = lib::FOO;
}
