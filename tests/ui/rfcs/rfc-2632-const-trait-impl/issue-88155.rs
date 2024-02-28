//@ known-bug: #110395

#![feature(const_trait_impl)]

pub trait A {
    fn assoc() -> bool;
}

pub const fn foo<T: A>() -> bool {
    T::assoc()
    //FIXME ~^ ERROR the trait
    //FIXME ~| ERROR cannot call non-const fn
}

fn main() {}
