#![feature(const_trait_impl)]

pub trait A {
    fn assoc() -> bool;
}

pub const fn foo<T: A>() -> bool {
    T::assoc()
    //~^ ERROR cannot call non-const associated function
}

fn main() {}
