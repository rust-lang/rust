#![allow(unused)]
#![feature(const_fn_trait_bound, const_trait_impl, inline_const)]

const fn f<T: ~const Drop>(x: T) {}

struct UnconstDrop;

impl Drop for UnconstDrop {
    fn drop(&mut self) {}
}

fn main() {
    const {
        f(UnconstDrop);
        //~^ ERROR the trait bound `UnconstDrop: Drop` is not satisfied
    }
}
