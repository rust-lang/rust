//! Regression test for <https://github.com/rust-lang/rust/issues/159169>.
//@ revisions: regular always_gca
//@[always_gca] run-pass
#![feature(gca_min_const_items, supertrait_item_shadowing)]
#![allow(dead_code)]

trait A {
    type Assoc;
    const CONST: i32;
}
impl<T> A for T {
    type Assoc = i8;
    const CONST: i32 = 1;
}

trait B: A {
    type Assoc;
    #[cfg_attr(always_gca, rustc_always_gca)]
    const CONST: i32;
}
impl<T> B for T {
    type Assoc = i16;
    const CONST: i32 = core::gca!(2);
    //[regular]~^ ERROR implementation of a regular const cannot have a `gca!` RHS
}

fn konst<T>() -> i32 {
    T::CONST
}

fn main() {
    assert_eq!(konst::<()>(), 2)
}
