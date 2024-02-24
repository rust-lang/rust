//@ revisions: rpass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn test<const SIZE: usize>() {}

trait SomeTrait {
    const SIZE: usize;
}

struct A<'a, T> {
    some_ref: &'a str,
    _maker: core::marker::PhantomData<T>,
}

impl<'a, T: SomeTrait> A<'a, T>
where
    [(); T::SIZE]: ,
{
    fn call_test() {
        test::<{ T::SIZE }>();
    }
}

fn main() {}
