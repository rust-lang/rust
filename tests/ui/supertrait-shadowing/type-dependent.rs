//@ run-pass

// Makes sure we can shadow with type-dependent associated item syntax.

#![feature(min_generic_const_args)]
#![feature(supertrait_item_shadowing)]
#![allow(dead_code)]

use std::mem::size_of;

trait A {
    fn hello() -> &'static str {
        "A"
    }
    type Assoc;
    const CONST: i32;
}
impl<T> A for T {
    type Assoc = i8;
    const CONST: i32 = 1;
}

trait B: A {
    fn hello() -> &'static str {
        "B"
    }
    type Assoc;
    type const CONST: i32;
}
impl<T> B for T {
    type Assoc = i16;
    type const CONST: i32 = 2;
}

fn foo<T>() -> &'static str {
    T::hello()
}

fn assoc<T: B>() -> usize {
    size_of::<T::Assoc>()
}

fn konst<T: B>() -> i32 {
    T::CONST
}

fn main() {
    assert_eq!(foo::<()>(), "B");
    assert_eq!(assoc::<()>(), 2);
    assert_eq!(konst::<()>(), 2);
}
