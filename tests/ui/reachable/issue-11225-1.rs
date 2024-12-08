//@ run-pass
//@ aux-build:issue-11225-1.rs


extern crate issue_11225_1 as foo;

pub fn main() {
    foo::foo(1);
    foo::foo_ufcs(1);
}
