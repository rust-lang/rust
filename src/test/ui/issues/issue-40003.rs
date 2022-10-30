// run-pass
#![allow(unused_must_use)]
fn main() {
    if false { test(); }
}

fn test() {
    let rx = Err::<Vec<usize>, u32>(1).into_future();

    rx.map(|l: Vec<usize>| stream::iter(l.into_iter().map(|i| Ok(i))))
      .flatten_stream()
      .chunks(50)
      .buffer_unordered(5);
}

use future::{Future, IntoFuture};
mod future {
    use std::result;

    use {stream, Stream};

    pub trait Future {
        type Item;
        type Error;

        fn map<F, U>(self, _: F) -> Map<Self, F>
            where F: FnOnce(Self::Item) -> U,
                  Self: Sized,
        {
            panic!()
        }

        fn flatten_stream(self) -> FlattenStream<Self>
            where <Self as Future>::Item: stream::Stream<Error=Self::Error>,
                  Self: Sized
        {
            panic!()
        }
    }

    pub trait IntoFuture {
        type Future: Future<Item=Self::Item, Error=Self::Error>;
        type Item;
        type Error;
        fn into_future(self) -> Self::Future;
    }

    impl<F: Future> IntoFuture for F {
        type Future = F;
        type Item = F::Item;
        type Error = F::Error;

        fn into_future(self) -> F {
            panic!()
        }
    }

    impl<T, E> IntoFuture for result::Result<T, E> {
        type Future = FutureResult<T, E>;
        type Item = T;
        type Error = E;

        fn into_future(self) -> FutureResult<T, E> {
            panic!()
        }
    }

    pub struct Map<A, F> {
        _a: (A, F),
    }

    impl<U, A, F> Future for Map<A, F>
        where A: Future,
              F: FnOnce(A::Item) -> U,
    {
        type Item = U;
        type Error = A::Error;
    }

    pub struct FlattenStream<F> {
        _f: F,
    }

    impl<F> Stream for FlattenStream<F>
        where F: Future,
              <F as Future>::Item: Stream<Error=F::Error>,
    {
        type Item = <F::Item as Stream>::Item;
        type Error = <F::Item as Stream>::Error;
    }

    pub struct FutureResult<T, E> {
        _inner: (T, E),
    }

    impl<T, E> Future for FutureResult<T, E> {
        type Item = T;
        type Error = E;
    }
}

mod stream {
    use IntoFuture;

    pub trait Stream {
        type Item;
        type Error;

        fn buffer_unordered(self, amt: usize) -> BufferUnordered<Self>
            where Self::Item: IntoFuture<Error = <Self as Stream>::Error>,
                  Self: Sized
        {
            new(self, amt)
        }

        fn chunks(self, _capacity: usize) -> Chunks<Self>
            where Self: Sized
        {
            panic!()
        }
    }

    pub struct IterStream<I> {
        _iter: I,
    }

    pub fn iter<J, T, E>(_: J) -> IterStream<J::IntoIter>
        where J: IntoIterator<Item=Result<T, E>>,
    {
        panic!()
    }

    impl<I, T, E> Stream for IterStream<I>
        where I: Iterator<Item=Result<T, E>>,
    {
        type Item = T;
        type Error = E;
    }

    pub struct Chunks<S> {
        _stream: S
    }

    impl<S> Stream for Chunks<S>
        where S: Stream
    {
        type Item = Result<Vec<<S as Stream>::Item>, u32>;
        type Error = <S as Stream>::Error;
    }

    pub struct BufferUnordered<S> {
        _stream: S,
    }

    enum Slot<T> {
        Next(#[allow(unused_tuple_struct_fields)] usize),
        _Data { _a: T },
    }

    fn new<S>(_s: S, _amt: usize) -> BufferUnordered<S>
        where S: Stream,
              S::Item: IntoFuture<Error=<S as Stream>::Error>,
    {
        (0..0).map(|_| {
            Slot::Next::<<S::Item as IntoFuture>::Future>(1)
        }).collect::<Vec<_>>();
        panic!()
    }

    impl<S> Stream for BufferUnordered<S>
        where S: Stream,
              S::Item: IntoFuture<Error=<S as Stream>::Error>,
    {
        type Item = <S::Item as IntoFuture>::Item;
        type Error = <S as Stream>::Error;
    }
}
use stream::Stream;
