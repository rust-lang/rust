//@ run-pass

#![feature(supertrait_item_shadowing)]
#![feature(min_generic_const_args)]
#![warn(resolving_to_items_shadowing_supertrait_items)]
#![warn(shadowing_supertrait_items)]
#![allow(dead_code)]

use std::mem::size_of;

trait A {
    fn hello(&self) -> &'static str {
        "A"
    }
    type Assoc;
    const CONST: i32;
}
impl<T> A for T {
    type Assoc = i8;
    const CONST: i32 = 1;
}

trait B {
    fn hello(&self) -> &'static str {
        "B"
    }
    type Assoc;
    const CONST: i32;
}
impl<T> B for T {
    type Assoc = i16;
    const CONST: i32 = 2;
}

trait C: A + B {
    fn hello(&self) -> &'static str {
        //~^ WARN trait item `hello` from `C` shadows identically named item
        "C"
    }
    type Assoc;
    //~^ WARN trait item `Assoc` from `C` shadows identically named item
    type const CONST: i32;
    //~^ WARN trait item `CONST` from `C` shadows identically named item
}
impl<T> C for T {
    type Assoc = i32;
    type const CONST: i32 = 3;
}

fn main() {
    assert_eq!(().hello(), "C");
    //~^ WARN trait item `hello` from `C` shadows identically named item from supertrait
    check::<()>();
}

fn check<T: C>() {
    assert_eq!(size_of::<T::Assoc>(), 4);
    assert_eq!(T::CONST, 3);
}
