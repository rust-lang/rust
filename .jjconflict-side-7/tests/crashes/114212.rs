//@ known-bug: #114212

#![feature(generic_const_exprs)]

use core::marker::PhantomData;

pub const DEFAULT_MAX_INPUT_LEN: usize = 256;

pub trait FooTrait {}

pub struct Foo<const MAX_INPUT_LEN: usize>;

impl<const MAX_INPUT_LEN: usize> FooTrait for Foo<MAX_INPUT_LEN> {}

pub struct Bar<
    const MAX_INPUT_LEN: usize = DEFAULT_MAX_INPUT_LEN,
    PB = Foo<MAX_INPUT_LEN>,
>
where
    PB: FooTrait,
{
    _pb: PhantomData<PB>,
}

impl<const MAX_INPUT_LEN: usize, PB> Bar<MAX_INPUT_LEN, PB>
where
    PB: FooTrait,
{
    pub fn new() -> Self {
        Self {
            _pb: PhantomData,
        }
    }
}
