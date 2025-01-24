//@ run-pass
//@ aux-build:issue-11225-2.rs


extern crate issue_11225_2 as foo;

pub fn main() {
    foo::foo(1);
    foo::foo_ufcs(1);
}
