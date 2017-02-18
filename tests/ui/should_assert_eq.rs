#![feature(plugin)]
#![plugin(clippy)]

#![allow(needless_take_by_value)]
#![deny(should_assert_eq)]

#[derive(PartialEq, Eq)]
struct NonDebug(i32);

#[derive(Debug, PartialEq, Eq)]
struct Debug(i32);

fn main() {
    assert!(1 == 2);
    assert!(Debug(1) == Debug(2));
    assert!(NonDebug(1) == NonDebug(1)); // ok

    test_generic(1, 2, 3, 4);
}

fn test_generic<T: std::fmt::Debug + Eq, U: Eq>(x: T, y: T, z: U, w: U) {
    assert!(x == y);
    assert!(z == w); // ok
}
