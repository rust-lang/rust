//@ run-pass
//@ aux-build:issue-9155.rs


extern crate issue_9155;

struct Baz;

pub fn main() {
    issue_9155::Foo::new(Baz);
}

// https://github.com/rust-lang/rust/issues/9155
