// compile-flags: -Znormalize-docs

use std::ops::Index;

pub fn next<'a, T>(s: &'a mut dyn SVec<Item = T, Output = T>) {
    let _ = s;
}

pub trait SVec: Index<
    <Self as SVec>::Item,
    Output = <Index<<Self as SVec>::Item,
    Output = <Self as SVec>::Item> as SVec>::Item,
> {
    type Item<'a, T>;

    fn len(&self) -> <Self as SVec>::Item;
    //~^ ERROR
    //~^^ ERROR
}
