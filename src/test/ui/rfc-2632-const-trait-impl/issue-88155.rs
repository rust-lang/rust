#![feature(const_fn_trait_bound)]
#![feature(const_trait_impl)]

pub trait A {
    fn assoc() -> bool;
}

pub const fn foo<T: A>() -> bool {
    T::assoc()
    //~^ ERROR calls in constant functions are limited
}

fn main() {}
