//@ known-bug: #119729
#![feature(generic_const_exprs)]

trait Size<const N: usize> {}

impl<T: Sized> Size<{ std::mem::size_of::<T>() }> for T {}

struct A<T: Size<8> + ?Sized> {
    x: std::marker::PhantomData<T>,
}

fn foo(x: A<dyn Send>) {}
