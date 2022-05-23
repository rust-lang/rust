#![allow(unused)]
#![feature(const_trait_impl, inline_const, negative_impls)]

use std::marker::Destruct;

const fn f<T: ~const Destruct>(x: T) {}

struct UnconstDrop;

impl Drop for UnconstDrop {
    fn drop(&mut self) {}
}

struct NonDrop;

impl !Drop for NonDrop {}

fn main() {
    const {
        f(UnconstDrop);
        //~^ ERROR can't drop
        f(NonDrop);
        //~^ ERROR can't drop
    }
}
