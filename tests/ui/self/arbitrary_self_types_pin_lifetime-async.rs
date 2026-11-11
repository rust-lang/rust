//@ check-pass
//@ edition:2018

use std::pin::Pin;
use std::task::{Context, Poll};

struct Foo;

impl Foo {
    async fn pin_ref(self: Pin<&Self>) -> Pin<&Self> { self }

    async fn pin_mut(self: Pin<&mut Self>) -> Pin<&mut Self> { self }

    async fn pin_pin_pin_ref(self: Pin<Pin<Pin<&Self>>>) -> Pin<Pin<Pin<&Self>>> { self }

    async fn pin_ref_impl_trait(self: Pin<&Self>) -> impl Clone + '_ { self }

    fn b(self: Pin<&Foo>, f: &Foo) -> Pin<&Foo> { self }
}

type Alias<T> = Pin<T>;
impl Foo {
    async fn bar<'a>(self: Alias<&Self>, arg: &'a ()) -> Alias<&Self> { self }
}

// FIXME(Centril): extend with the rest of the non-`async fn` test
// when we allow `async fn`s inside traits and trait implementations.

fn main() {
    let mut foo = Foo;
    { Pin::new(&foo).pin_ref() };
    { Pin::new(&mut foo).pin_mut() };
    { Pin::new(Pin::new(Pin::new(&foo))).pin_pin_pin_ref() };
    { Pin::new(&foo).pin_ref_impl_trait() };
}
