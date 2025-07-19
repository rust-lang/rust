// Repro for <https://github.com/rust-lang/rust/issues/90696#issuecomment-963375847>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] check-pass
//@[no_assumptions] known-bug: #110338

#![allow(dropping_copy_types)]

use std::{future::Future, marker::PhantomData};

trait Trait {
    type Associated<'a>: Send
    where
        Self: 'a;
}

fn future<'a, S: Trait + 'a, F>(f: F) -> F
where
    F: Future<Output = ()> + Send,
{
    f
}

fn foo<'a, S: Trait + 'a>() {
    future::<'a, S, _>(async move {
        let result: PhantomData<S::Associated<'a>> = PhantomData;
        async {}.await;
        drop(result);
    });
}

fn main() {}
