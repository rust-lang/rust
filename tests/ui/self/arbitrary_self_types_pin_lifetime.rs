//@ check-pass

use std::pin::Pin;
use std::task::{Context, Poll};

struct Foo;

impl Foo {
    fn pin_ref(self: Pin<&Self>) -> Pin<&Self> { self }

    fn pin_mut(self: Pin<&mut Self>) -> Pin<&mut Self> { self }

    fn pin_pin_pin_ref(self: Pin<Pin<Pin<&Self>>>) -> Pin<Pin<Pin<&Self>>> { self }

    fn pin_ref_impl_trait(self: Pin<&Self>) -> impl Clone + '_ { self }

    fn b(self: Pin<&Foo>, f: &Foo) -> Pin<&Foo> { self }
}

type Alias<T> = Pin<T>;
impl Foo {
    fn bar<'a>(self: Alias<&Self>, arg: &'a ()) -> Alias<&Self> { self }
}

struct Bar<T: Unpin, U: Unpin> {
    field1: T,
    field2: U,
}

impl<T: Unpin, U: Unpin> Bar<T, U> {
    fn fields(self: Pin<&mut Self>) -> (Pin<&mut T>, Pin<&mut U>) {
        let this = self.get_mut();
        (Pin::new(&mut this.field1), Pin::new(&mut this.field2))
    }
}

trait AsyncBufRead {
    fn poll_fill_buf(self: Pin<&mut Self>, cx: &mut Context<'_>)
        -> Poll<std::io::Result<&[u8]>>;
}

struct Baz(Vec<u8>);

impl AsyncBufRead for Baz {
    fn poll_fill_buf(self: Pin<&mut Self>, cx: &mut Context<'_>)
        -> Poll<std::io::Result<&[u8]>>
    {
        Poll::Ready(Ok(&self.get_mut().0))
    }
}

fn main() {
    let mut foo = Foo;
    { Pin::new(&foo).pin_ref() };
    { Pin::new(&mut foo).pin_mut() };
    { Pin::new(Pin::new(Pin::new(&foo))).pin_pin_pin_ref() };
    { Pin::new(&foo).pin_ref_impl_trait() };
    let mut bar = Bar { field1: 0u8, field2: 1u8 };
    { Pin::new(&mut bar).fields() };
}
