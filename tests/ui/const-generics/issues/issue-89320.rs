// build-pass

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub trait Enumerable {
    const N: usize;
}

#[derive(Clone)]
pub struct SymmetricGroup<S>
where
    S: Enumerable,
    [(); S::N]: Sized,
{
    _phantom: std::marker::PhantomData<S>,
}

fn main() {}
