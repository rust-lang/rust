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
    generic3::<u32>();
    generic4::<u32>();
}

fn generic<U: B>() {
    println!("{}", size_of::<U::T>());
}

fn generic2<U: A<T = i8>>() {
    println!("{}", size_of::<U::T>());
}

fn generic3<U: B<T = i16>>() {
    println!("{}", size_of::<U::T>());
}

fn generic4<U: C<T = i16>>() {
    println!("{}", size_of::<U::T>());
}
