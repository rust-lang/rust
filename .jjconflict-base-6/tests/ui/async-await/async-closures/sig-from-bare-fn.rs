//@ check-pass
//@ edition: 2021

// Make sure that we infer the args of an async closure even if it's passed to
// a function that requires the async closure implement `Fn*` but does *not* have
// a `Future` bound on the return type.

use std::future::Future;

trait TryStream {
    type Ok;
    type Err;
}

trait TryFuture {
    type Ok;
    type Err;
}

impl<F, T, E> TryFuture for F where F: Future<Output = Result<T, E>> {
    type Ok = T;
    type Err = E;
}

trait TryStreamExt: TryStream {
    fn try_for_each<F, Fut>(&self, f: F)
    where
        F: FnMut(Self::Ok) -> Fut,
        Fut: TryFuture<Ok = (), Err = Self::Err>;
}

impl<S> TryStreamExt for S where S: TryStream {
    fn try_for_each<F, Fut>(&self, f: F)
    where
        F: FnMut(Self::Ok) -> Fut,
        Fut: TryFuture<Ok = (), Err = Self::Err>,
    { }
}

fn test(stream: impl TryStream<Ok = &'static str, Err = ()>) {
    stream.try_for_each(async |s| {
        s.trim(); // Make sure we know the type of `s` at this point.
        Ok(())
    });
}

fn main() {}
