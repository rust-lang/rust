#![feature(supertrait_item_shadowing)]
#![allow(dead_code)]

use std::mem::size_of;

trait A {
    type Assoc;
}
impl<X> A for X {
    type Assoc = i8;
}

trait B: A {
    type Assoc;
}
impl<X> B for X {
    type Assoc = i16;
}

trait C: B {}
impl<X> C for X {}

fn main() {
    b_unbound::<u32>();
    c_unbound::<u32>();

    b_assoc_is_a::<u32>();
    //~^ ERROR type mismatch resolving `<u32 as B>::Assoc == i8`
    c_assoc_is_a::<u32>();
    //~^ ERROR type mismatch resolving `<u32 as B>::Assoc == i8`
}

fn b_unbound<U: B>() {
    let _ = size_of::<U::Assoc>();
}

fn c_unbound<U: C>() {
    let _ = size_of::<U::Assoc>();
}

fn b_assoc_is_a<U: B<Assoc = i8>>() {
    let _ = size_of::<U::Assoc>();
}

fn c_assoc_is_a<U: C<Assoc = i8>>() {
    let _ = size_of::<U::Assoc>();
}
