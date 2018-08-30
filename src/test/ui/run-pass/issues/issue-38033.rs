// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::marker;
use std::mem;

fn main() {
    let workers = (0..0).map(|_| result::<u32, ()>());
    drop(join_all(workers).poll());
}

trait Future {
    type Item;
    type Error;

    fn poll(&mut self) -> Result<Self::Item, Self::Error>;
}

trait IntoFuture {
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
        self
    }
}

struct FutureResult<T, E> {
    _inner: marker::PhantomData<(T, E)>,
}

fn result<T, E>() -> FutureResult<T, E> {
    loop {}
}

impl<T, E> Future for FutureResult<T, E> {
    type Item = T;
    type Error = E;

    fn poll(&mut self) -> Result<T, E> {
        loop {}
    }
}

struct JoinAll<I>
    where I: IntoIterator,
          I::Item: IntoFuture,
{
    elems: Vec<<I::Item as IntoFuture>::Item>,
}

fn join_all<I>(_: I) -> JoinAll<I>
    where I: IntoIterator,
          I::Item: IntoFuture,
{
    JoinAll { elems: vec![] }
}

impl<I> Future for JoinAll<I>
    where I: IntoIterator,
          I::Item: IntoFuture,
{
    type Item = Vec<<I::Item as IntoFuture>::Item>;
    type Error = <I::Item as IntoFuture>::Error;

    fn poll(&mut self) -> Result<Self::Item, Self::Error> {
        let elems = mem::replace(&mut self.elems, Vec::new());
        Ok(elems.into_iter().map(|e| {
            e
        }).collect::<Vec<_>>())
    }
}
