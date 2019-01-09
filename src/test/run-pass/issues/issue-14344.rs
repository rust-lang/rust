// run-pass
// aux-build:issue-14344-1.rs
// aux-build:issue-14344-2.rs

extern crate issue_14344_1;
extern crate issue_14344_2;

fn main() {
    issue_14344_1::foo();
    issue_14344_2::bar();
}
