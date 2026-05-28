//@ run-pass
#![allow(dead_code)]
// Test that pointers to extern types can be cast from/to usize,
// despite being !Sized.
#![feature(extern_types, sized_hierarchy)]
use std::marker::PointeeSized;

extern "C" {
    type A;
}

struct Foo {
    x: u8,
    tail: A,
}

struct Bar<T: PointeeSized> {
    x: u8,
    tail: T,
}

#[cfg(target_pointer_width = "32")]
const MAGIC: usize = 0xdeadbeef;
#[cfg(target_pointer_width = "64")]
const MAGIC: usize = 0x12345678deadbeef;

fn main() {
    assert_eq!((MAGIC as *const A) as usize, MAGIC);
    assert_eq!((MAGIC as *const Foo) as usize, MAGIC);
    assert_eq!((MAGIC as *const Bar<A>) as usize, MAGIC);
    assert_eq!((MAGIC as *const Bar<Bar<A>>) as usize, MAGIC);
}
