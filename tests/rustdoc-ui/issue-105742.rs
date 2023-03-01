// compile-flags: -Znormalize-docs
use std::ops::Index;

pub fn next<'a, T>(s: &'a mut dyn SVec<Item = T, Output = T>) {
    //~^ERROR
    //~|ERROR
    //~|ERROR
    let _ = s;
}

pub trait SVec: Index<
    <Self as SVec>::Item,
    //~^ERROR
    //~|ERROR
    //~|ERROR
    //~|ERROR
    Output = <Index<<Self as SVec>::Item,
    //~^ERROR
    //~|ERROR
    //~|ERROR
    //~|ERROR
    Output = <Self as SVec>::Item> as SVec>::Item,
    //~^ERROR
    //~|ERROR
    //~|ERROR
    //~|ERROR
    //~|ERROR
    //~|ERROR
    //~|ERROR
    //~|ERROR
> {
    type Item<'a, T>;

    fn len(&self) -> <Self as SVec>::Item;
    //~^ERROR
    //~|ERROR
}
