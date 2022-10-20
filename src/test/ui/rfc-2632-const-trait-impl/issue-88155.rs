#![feature(const_trait_impl, effects)]

pub trait A {
    fn assoc() -> bool;
}

pub const fn foo<T: A>() -> bool {
    T::assoc()
    //~^ ERROR cannot call non-const fn `<T as A>::assoc` in functions
}

fn main() {}
