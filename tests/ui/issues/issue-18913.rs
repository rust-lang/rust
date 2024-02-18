//@ run-pass
//@ aux-build:issue-18913-1.rs
//@ aux-build:issue-18913-2.rs

extern crate foo;

fn main() {
    assert_eq!(foo::foo(), 1);
}
