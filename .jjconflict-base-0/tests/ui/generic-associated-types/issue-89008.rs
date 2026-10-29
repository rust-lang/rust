//@ check-pass
//@ edition:2021

#![feature(impl_trait_in_assoc_type)]

use std::future::Future;
use std::marker::PhantomData;

trait Stream {
    type Item;
}

struct Empty<T> {
    _phantom: PhantomData<T>,
}

impl<T> Stream for Empty<T> {
    type Item = T;
}

trait X {
    type LineStream<'a, Repr>: Stream<Item = Repr> where Self: 'a;
    type LineStreamFut<'a, Repr>: Future<Output = Self::LineStream<'a, Repr>> where Self: 'a;
    fn line_stream<'a, Repr>(&'a self) -> Self::LineStreamFut<'a, Repr>;
}

struct Y;

impl X for Y {
    type LineStream<'a, Repr> = impl Stream<Item = Repr>;
    type LineStreamFut<'a, Repr> = impl Future<Output = Self::LineStream<'a, Repr>>;
    fn line_stream<'a, Repr>(&'a self) -> Self::LineStreamFut<'a, Repr> {
        async { Empty { _phantom: PhantomData } }
    }
}

fn main() {}
