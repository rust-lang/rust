// run-pass
// aux-build:issue-13872-1.rs
// aux-build:issue-13872-2.rs
// aux-build:issue-13872-3.rs

// pretty-expanded FIXME #23616

extern crate issue_13872_3 as other;

fn main() {
    other::foo();
}
