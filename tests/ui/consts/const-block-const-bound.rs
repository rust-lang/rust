// known-bug: #103507

#![allow(unused)]
#![feature(const_trait_impl, inline_const, negative_impls)]

use std::marker::Destruct;

const fn f<T: ~const Destruct>(x: T) {}

struct UnconstDrop;

impl Drop for UnconstDrop {
    fn drop(&mut self) {}
}

fn main() {
    const {
        f(UnconstDrop);
        //FIXME ~^ ERROR can't drop
    }
}
