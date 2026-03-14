//@ run-pass
//@ check-run-results

#![feature(supertrait_item_shadowing)]
#![allow(dead_code)]

use std::mem::size_of;

trait A {
    type T;
}
impl<T> A for T {
    type T = i8;
}

trait B: A {
    type T;
}
impl<T> B for T {
    type T = i16;
}

trait C: B {}
impl<T> C for T {}

fn main() {
    generic::<u32>();
    generic2::<u32>();
}

fn generic<T: B>() {
    println!("{}", size_of::<T::T>());
}

fn generic2<T: C<T = i16>>() {
    println!("{}", size_of::<T::T>());
}
