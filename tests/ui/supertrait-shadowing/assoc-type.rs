//@ run-pass
//@ check-run-results

#![feature(supertrait_item_shadowing)]
#![allow(dead_code)]

use std::mem::size_of;

trait A {
    type Assoc;
}
impl<T> A for T {
    type Assoc = i8;
}

trait B: A {
    type Assoc;
}
impl<T> B for T {
    type Assoc = i16;
}

trait C: B {}
impl<T> C for T {}

fn main() {
    generic::<u32>();
    generic2::<u32>();
    generic3::<u32>();
    generic4::<u32>();
    generic5::<u32>();
}

fn generic<T: B>() {
    assert_eq!(size_of::<T::Assoc>(), 2);
}

fn generic2<T: A<Assoc = i8>>() {
    assert_eq!(size_of::<T::Assoc>(), 1);
}

fn generic3<T: B<Assoc = i16>>() {
    assert_eq!(size_of::<T::Assoc>(), 2);
}

fn generic4<T: C<Assoc = i16>>() {
    assert_eq!(size_of::<T::Assoc>(), 2);
}

fn generic5<T: B>() {
    assert_eq!(size_of::<<T as A>::Assoc>(), 1);
    assert_eq!(size_of::<<T as B>::Assoc>(), 2);
}
