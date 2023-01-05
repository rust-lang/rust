#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Trait<T> {
    fn fnc<const N: usize = "">(&self) {} //~ERROR defaults for const parameters are only allowed in `struct`, `enum`, `type`, or `trait` definitions
    fn foo<const N: usize = { std::mem::size_of::<T>() }>(&self) {} //~ERROR defaults for const parameters are only allowed in `struct`, `enum`, `type`, or `trait` definitions
}

fn main() {}
