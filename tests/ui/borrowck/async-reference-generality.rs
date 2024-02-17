//@ check-pass
//@ edition: 2021

use std::marker::PhantomData;

pub struct Struct<I, T>(PhantomData<fn() -> <Self as It>::Item>)
where
    Self: It;

impl<I> It for Struct<I, I::Item>
where
    I: It,
{
    type Item = ();
}

pub trait It {
    type Item;
}

fn f() -> impl Send {
    async {
        let _x = Struct::<Empty<&'static ()>, _>(PhantomData);
        async {}.await;
    }
}

pub struct Empty<T>(PhantomData<fn() -> T>);

impl<T> It for Empty<T> {
    type Item = T;
}

fn main() {}
