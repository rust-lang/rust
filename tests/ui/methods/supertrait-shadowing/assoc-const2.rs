//@ run-pass
//@ check-run-results

#![feature(supertrait_item_shadowing)]
#![feature(min_generic_const_args)]
#![allow(dead_code)]

trait A {
    const CONST: i32;
}
impl<T> A for T {
    const CONST: i32 = 1;
}

trait B: A {
    type const CONST: i32;
}
impl<T> B for T {
    type const CONST: i32 = 2;
}

trait C: B {}
impl<T> C for T {}

fn main() {
    println!("{}", i32::CONST);
    generic::<u32>();
}

fn generic<T: C<CONST = 2>>() {
    println!("{}", T::CONST);
}
