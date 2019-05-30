#![allow(dead_code)]

#![feature(core_intrinsics)]

use std::intrinsics::type_name;

struct Foo<T> {
    x: T
}

pub fn main() {
    unsafe {
        assert_eq!(type_name::<isize>(), "isize");
        assert_eq!(type_name::<Foo<usize>>(), "tydesc_name::Foo<usize>");
    }
}
