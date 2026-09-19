//@ check-pass
//@ compile-flags: -Znext-solver=globally -Zrenormalize-rigid-aliases

#![allow(dead_code)]

trait Call<T> {}

trait Iter {
    type Item;
    fn apply<F>() where F: Call<Self::Item>;
}

trait Into {
    type Item;
    type Into: Iter<Item = Self::Item>;
}

impl<I: Iter> Into for I {
    type Item = I::Item;
    type Into = I;
}

struct Flatten<I>(I);

impl<I, U> Iter for Flatten<I>
where
    I: Iter<Item: Into<Into = U, Item = U::Item>>,
    U: Iter,
{
    type Item = U::Item;

    fn apply<F>() where F: Call<Self::Item> {}
}

fn implied<C: Into<Into = T>, T: Iter>(value: T::Item) -> C::Item {
    value
}

fn explicit<C: Into<Into = T, Item = T::Item>, T: Iter>(value: T::Item) -> C::Item {
    value
}

trait Identity {
    type Output;
}

trait Carrier {
    type Assoc: Identity;
}

impl Identity for u32 {
    type Output = Self;
}

fn concrete<C: Carrier<Assoc = u32>>(value: u32) -> <u32 as Identity>::Output {
    value
}

struct Borrowed<'a>(&'a ());

impl<'a> Identity for Borrowed<'a> {
    type Output = u32;
}

fn concrete_with_lifetime<'a, C: Carrier<Assoc = Borrowed<'a>>>(
    value: u32,
) -> <Borrowed<'a> as Identity>::Output {
    value
}

fn main() {}
