//@ check-pass

#![feature(precise_capturing_in_traits)]

struct Invariant<T>(*mut T);

trait Foo {
    fn hello<'s: 's>(&'s self) -> Invariant<impl Sized + use<Self>>;
}

fn hello<'s, T: Foo + 'static>(x: &'s T) -> Invariant<impl Sized + 'static> {
    x.hello()
}

fn main() {}
