//@ run-pass
//@ aux-build:issue-13620-1.rs
//@ aux-build:issue-13620-2.rs


extern crate issue_13620_2 as crate2;

fn main() {
    (crate2::FOO2.foo)();
}
