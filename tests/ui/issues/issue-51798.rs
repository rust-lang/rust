//@ edition:2018
//@ aux-build:issue-51798.rs
//@ check-pass

extern crate issue_51798;

mod server {
    fn f() {
        let mut v = issue_51798::vec();
        v.clear();
    }
}

fn main() {}
