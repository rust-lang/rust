// compile-flags: -Znormalize-docs
use std::ops::Index;

pub fn next<'a, T>(s: &'a mut dyn SVec<Item = T, Output = T>) {
    //~^ expected 1 lifetime argument
    //~| expected 1 generic argument
    //~| the trait `SVec` cannot be made into an object
    //~| `SVec` cannot be made into an object
    //~| missing generics for associated type `SVec::Item`
    //~| missing generics for associated type `SVec::Item`
    let _ = s;
}

pub trait SVec: Index<
    <Self as SVec>::Item,
    //~^ expected 1 lifetime argument
    //~| expected 1 generic argument
    //~| missing generics for associated type `SVec::Item`
    //~| missing generics for associated type `SVec::Item`
    //~| missing generics for associated type `SVec::Item`
    //~| missing generics for associated type `SVec::Item`
    Output = <Index<<Self as SVec>::Item,
    //~^ expected 1 lifetime argument
    //~| expected 1 generic argument
    //~| missing generics for associated type `SVec::Item`
    //~| missing generics for associated type `SVec::Item`
    //~| missing generics for associated type `SVec::Item`
    //~| missing generics for associated type `SVec::Item`
    Output = <Self as SVec>::Item> as SVec>::Item,
    //~^ expected 1 lifetime argument
    //~| expected 1 generic argument
    //~| expected 1 lifetime argument
    //~| missing generics for associated type `SVec::Item`
    //~| missing generics for associated type `SVec::Item`
    //~| missing generics for associated type `SVec::Item`
    //~| missing generics for associated type `SVec::Item`
    //~| expected 1 generic argument
    //~| missing generics for associated type `SVec::Item`
    //~| missing generics for associated type `SVec::Item`
    //~| missing generics for associated type `SVec::Item`
    //~| missing generics for associated type `SVec::Item`
> {
    type Item<'a, T>;

    fn len(&self) -> <Self as SVec>::Item;
    //~^ expected 1 lifetime argument
    //~| missing generics for associated type `SVec::Item`
    //~| expected 1 generic argument
    //~| missing generics for associated type `SVec::Item`
}
