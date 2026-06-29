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

trait B: A {
    fn hello(&self) -> &'static str {
        //~^ WARN trait item `hello` from `B` shadows identically named item
        "B"
    }
    type Assoc;
    //~^ WARN trait item `Assoc` from `B` shadows identically named item
    type const CONST: i32;
    //~^ WARN trait item `CONST` from `B` shadows identically named item
}
impl<T> B for T {
    type Assoc = i16;
    type const CONST: i32 = 2;
}

fn main() {
    assert_eq!(().hello(), "B");
    //~^ WARN trait item `hello` from `B` shadows identically named item from supertrait
    check::<()>();
}

fn check<T: B>() {
    assert_eq!(size_of::<T::Assoc>(), 2);
    assert_eq!(T::CONST, 2);
}
