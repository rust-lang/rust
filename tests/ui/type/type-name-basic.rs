//! Checks the basic functionality of `std::any::type_name` for primitive types
//! and simple generic structs.

//@ run-pass

#![allow(dead_code)]

use std::any::type_name;

struct Foo<T> {
    x: T,
}

pub fn main() {
    assert_eq!(type_name::<isize>(), "isize");
    assert_eq!(type_name::<Foo<usize>>(), "type_name_basic::Foo<usize>");
}
