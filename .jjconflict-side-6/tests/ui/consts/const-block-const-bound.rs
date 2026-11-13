#![allow(unused)]
#![feature(const_trait_impl, negative_impls, const_destruct)]

use std::marker::Destruct;

const fn f<T: [const] Destruct>(x: T) {}

struct UnconstDrop;

impl Drop for UnconstDrop {
    fn drop(&mut self) {}
}

fn main() {
    const {
        f(UnconstDrop);
        //~^ ERROR trait bound `UnconstDrop: const Destruct` is not satisfied
    }
}
