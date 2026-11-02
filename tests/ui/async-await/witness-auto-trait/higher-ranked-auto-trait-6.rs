// Repro for <https://github.com/rust-lang/rust/issues/82921#issue-825114180>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] check-pass
//@[no_assumptions] known-bug: #110338

use core::future::Future;
use core::marker::PhantomData;
use core::pin::Pin;
use core::task::{Context, Poll};

async fn f() {}

pub fn fail<'a>() -> Box<dyn Future<Output = ()> + Send + 'a> {
    Box::new(async { new(|| async { f().await }).await })
}

fn new<A, B>(_a: A) -> F<A, B>
where
    A: Fn() -> B,
{
    F { _i: PhantomData }
}

trait Stream {
    type Item;
}

struct T<A, B> {
    _a: PhantomData<A>,
    _b: PhantomData<B>,
}

impl<A, B> Stream for T<A, B>
where
    A: Fn() -> B,
{
    type Item = B;
}

struct F<A, B>
where
    A: Fn() -> B,
{
    _i: PhantomData<<T<A, B> as Stream>::Item>,
}

impl<A, B> Future for F<A, B>
where
    A: Fn() -> B,
{
    type Output = ();
    fn poll(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<Self::Output> {
        unimplemented!()
    }
}

fn main() {}
