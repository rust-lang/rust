//@ run-pass
//@ aux-build:issue-9906.rs


extern crate issue_9906 as testmod;

pub fn main() {
    testmod::foo();
    testmod::FooBar::new(1);
}
