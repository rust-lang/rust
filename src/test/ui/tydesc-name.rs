// run-pass

#![allow(dead_code)]

use std::any::type_name;

struct Foo<T> {
    x: T
}

pub fn main() {
    assert_eq!(type_name::<isize>(), "isize");
    assert_eq!(type_name::<Foo<usize>>(), "tydesc_name::Foo<usize>");
}
