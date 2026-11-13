//! Regression test for #119729

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Size<const N: usize> {}

impl<T: Sized> Size<{ std::mem::size_of::<T>() }> for T {}

struct A<T: Size<8> + ?Sized> {
    x: std::marker::PhantomData<T>,
}

fn foo(x: A<dyn Send>) {}
//~^ ERROR mismatched types
//~| ERROR the size for values of type `(dyn Send + 'static)` cannot be known at compilation time

fn main() {}
