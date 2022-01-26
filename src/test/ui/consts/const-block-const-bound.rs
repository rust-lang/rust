#![allow(unused)]
#![feature(const_fn_trait_bound, const_trait_impl, inline_const, negative_impls)]

const fn f<T: ~const Drop>(x: T) {}

struct UnconstDrop;

impl Drop for UnconstDrop {
    fn drop(&mut self) {}
}

struct NonDrop;

impl !Drop for NonDrop {}

fn main() {
    const {
        f(UnconstDrop);
        //~^ ERROR the trait bound `UnconstDrop: ~const Drop` is not satisfied
        f(NonDrop);
        //~^ ERROR the trait bound `NonDrop: ~const Drop` is not satisfied
    }
}
