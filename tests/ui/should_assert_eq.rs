#![feature(plugin)]
#![plugin(clippy)]

#![allow(needless_pass_by_value)]
#![warn(should_assert_eq)]

#[derive(PartialEq, Eq)]
struct NonDebug(i32);

#[derive(Debug, PartialEq, Eq)]
struct Debug(i32);

fn main() {
    assert!(1 == 2);
    assert!(Debug(1) == Debug(2));
    assert!(NonDebug(1) == NonDebug(1)); // ok
    assert!(Debug(1) != Debug(2));
    assert!(NonDebug(1) != NonDebug(2)); // ok

    test_generic(1, 2, 3, 4);

    debug_assert!(4 == 5);
    debug_assert!(4 != 6);
}

fn test_generic<T: std::fmt::Debug + Eq, U: Eq>(x: T, y: T, z: U, w: U) {
    assert!(x == y);
    assert!(z == w); // ok

    assert!(x != y);
    assert!(z != w); // ok
}
