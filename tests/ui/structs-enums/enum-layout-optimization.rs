//@ run-pass
// Test that we will do various size optimizations to enum layout, but
// *not* if `#[repr(u8)]` or `#[repr(C)]` is passed. See also #40029.

#![allow(dead_code)]

use std::mem;

enum Nullable<T> {
    Alive(T),
    Dropped,
}

#[repr(u8)]
enum NullableU8<T> {
    Alive(T),
    Dropped,
}

#[repr(C)]
enum NullableC<T> {
    Alive(T),
    Dropped,
}

struct StructNewtype<T>(T);

#[repr(C)]
struct StructNewtypeC<T>(T);

enum EnumNewtype<T> { Variant(T) }

#[repr(u8)]
enum EnumNewtypeU8<T> { Variant(T) }

#[repr(C)]
enum EnumNewtypeC<T> { Variant(T) }

fn main() {
    assert!(mem::size_of::<Box<i32>>() == mem::size_of::<Nullable<Box<i32>>>());
    assert!(mem::size_of::<Box<i32>>() < mem::size_of::<NullableU8<Box<i32>>>());
    assert!(mem::size_of::<Box<i32>>() < mem::size_of::<NullableC<Box<i32>>>());

    assert!(mem::size_of::<i32>() == mem::size_of::<StructNewtype<i32>>());
    assert!(mem::size_of::<i32>() == mem::size_of::<StructNewtypeC<i32>>());

    assert!(mem::size_of::<i32>() == mem::size_of::<EnumNewtype<i32>>());
    assert!(mem::size_of::<i32>() < mem::size_of::<EnumNewtypeU8<i32>>());
    assert!(mem::size_of::<i32>() < mem::size_of::<EnumNewtypeC<i32>>());
}
