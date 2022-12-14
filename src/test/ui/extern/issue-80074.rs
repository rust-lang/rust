// edition:2018
// build-pass
// aux-crate:issue_80074=issue-80074-macro.rs

#[macro_use]
extern crate issue_80074;

fn main() {
    foo!();
}
