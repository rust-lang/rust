//@ compile-flags: -Znormalize-docs
//@ dont-require-annotations: NOTE

// https://github.com/rust-lang/rust/issues/105742
use std::ops::Index;

pub fn next<'a, T>(s: &'a mut dyn SVec<Item = T, Output = T>) {
    //~^ NOTE expected 1 lifetime argument
    //~| NOTE expected 1 generic argument
    //~| ERROR the trait `SVec` is not dyn compatible
    //~| NOTE `SVec` is not dyn compatible
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    let _ = s;
}

pub trait SVec: Index<
    <Self as SVec>::Item,
    //~^ NOTE expected 1 lifetime argument
    //~| NOTE expected 1 generic argument
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    Output = <Index<<Self as SVec>::Item,
    //~^ NOTE expected 1 lifetime argument
    //~| NOTE expected 1 generic argument
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    Output = <Self as SVec>::Item> as SVec>::Item,
    //~^ NOTE expected 1 lifetime argument
    //~| NOTE expected 1 generic argument
    //~| NOTE expected 1 lifetime argument
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| NOTE expected 1 generic argument
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| ERROR missing generics for associated type `SVec::Item`
> {
    type Item<'a, T>;

    fn len(&self) -> <Self as SVec>::Item;
    //~^ NOTE expected 1 lifetime argument
    //~| ERROR missing generics for associated type `SVec::Item`
    //~| NOTE expected 1 generic argument
    //~| ERROR missing generics for associated type `SVec::Item`
}
