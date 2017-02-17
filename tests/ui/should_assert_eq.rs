#![feature(plugin)]
#![plugin(clippy)]

#![deny(should_assert_eq)]

#[derive(PartialEq, Eq)]
struct NonDebug(i32);

#[derive(Debug, PartialEq, Eq)]
struct Debug(i32);

fn main() {
    assert!(1 == 2);
    assert!(Debug(1) == Debug(2));
    assert!(NonDebug(1) == NonDebug(1)); // ok
}
