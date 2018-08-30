// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused)]

fn main() {
}

fn foo() {
    let b = mk::<
        Forward<(Box<Future<Error = u32>>,)>,
    >();
    b.map_err(|_| ()).join();
}

fn mk<T>() -> T {
    loop {}
}

impl<I: Future<Error = E>, E> Future for (I,) {
    type Error = E;
}

struct Forward<T: Future> {
    _a: T,
}

impl<T: Future> Future for Forward<T>
where
    T::Error: From<u32>,
{
    type Error = T::Error;
}

trait Future {
    type Error;

    fn map_err<F, E>(self, _: F) -> (Self, F)
    where
        F: FnOnce(Self::Error) -> E,
        Self: Sized,
    {
        loop {}
    }

    fn join(self) -> (MaybeDone<Self>, ())
    where
        Self: Sized,
    {
        loop {}
    }
}

impl<S: ?Sized + Future> Future for Box<S> {
    type Error = S::Error;
}

enum MaybeDone<A: Future> {
    _Done(A::Error),
}

impl<U, A: Future, F> Future for (A, F)
where
    F: FnOnce(A::Error) -> U,
{
    type Error = U;
}
