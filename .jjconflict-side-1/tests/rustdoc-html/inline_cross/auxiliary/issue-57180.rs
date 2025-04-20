//@ compile-flags: -Cmetadata=aux

pub trait Trait {
}

pub struct Struct<F>
{
    _p: ::std::marker::PhantomData<F>,
}

impl<F: Fn() -> u32>
Trait for Struct<F>
    where
        F: Fn() -> u32,
{
}
