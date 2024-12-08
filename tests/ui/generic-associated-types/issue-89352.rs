//@ check-pass

use std::marker::PhantomData;

pub trait GenAssoc<T> {
    type Iter<'at>;
    fn iter(&self) -> Self::Iter<'_>;
    fn reborrow<'longt: 'shortt, 'shortt>(iter: Self::Iter<'longt>) -> Self::Iter<'shortt>;
}

pub struct Wrapper<'a, T: 'a, A: GenAssoc<T>> {
    a: A::Iter<'a>,
    _p: PhantomData<T>,
}

impl<'ai, T: 'ai, A: GenAssoc<T>> GenAssoc<T> for Wrapper<'ai, T, A>
where
    A::Iter<'ai>: Clone,
{
    type Iter<'b> = ();
    fn iter<'s>(&'s self) -> Self::Iter<'s> {
        let a = A::reborrow::<'ai, 's>(self.a.clone());
    }

    fn reborrow<'long: 'short, 'short>(iter: Self::Iter<'long>) -> Self::Iter<'short> {
        ()
    }
}

fn main() {}
