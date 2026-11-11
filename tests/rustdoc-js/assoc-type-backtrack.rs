#![feature(rustdoc_internals)]

pub trait MyTrait2<X> {
    type Output;
}

pub trait MyTrait {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: MyTrait2<(B, Self::Item), Output = B>;
}

pub struct Cloned<I>(I);

impl<'a, T, I> MyTrait for Cloned<I>
where
    T: 'a + Clone,
    I: MyTrait<Item = &'a T>,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        loop {}
    }
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: MyTrait2<(B, Self::Item), Output = B>,
    {
        loop {}
    }
}

#[doc(search_unbox)]
pub trait MyFuture {
    type Output;
}

#[doc(search_unbox)]
pub trait MyIntoFuture {
    type Output;
    type Fut: MyFuture<Output = Self::Output>;
    fn into_future(self) -> Self::Fut;
    fn into_future_2(self, other: Self) -> Self::Fut;
}
