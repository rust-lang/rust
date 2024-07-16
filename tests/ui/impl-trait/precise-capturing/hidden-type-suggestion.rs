#![feature(precise_capturing)]

fn lifetime<'a, 'b>(x: &'a ()) -> impl Sized + use<'b> {
//~^ HELP add `'a` to the `use<...>` bound
    x
//~^ ERROR hidden type for
}

fn param<'a, T>(x: &'a ()) -> impl Sized + use<T> {
//~^ HELP add `'a` to the `use<...>` bound
    x
//~^ ERROR hidden type for
}

fn empty<'a>(x: &'a ()) -> impl Sized + use<> {
//~^ HELP add `'a` to the `use<...>` bound
    x
//~^ ERROR hidden type for
}

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

fn missing<'a, 'captured, 'not_captured, Captured>(x: &'a ()) -> impl Captures<'captured> {
//~^ HELP add a `use<...>` bound
    x
//~^ ERROR hidden type for
}

fn main() {}
