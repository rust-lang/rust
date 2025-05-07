//@ run-pass
// Test that we don't panic on a RefCell borrow conflict in certain
// code paths involving unboxed closures.


//@ aux-build:issue-18711.rs
extern crate issue_18711 as issue;

fn main() {
    (|| issue::inner(()))();
}
