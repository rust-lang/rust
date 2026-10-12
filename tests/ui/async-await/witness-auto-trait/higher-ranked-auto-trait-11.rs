// Repro for <https://github.com/rust-lang/rust/issues/60658#issuecomment-1509321859>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] known-bug: unknown
//@[no_assumptions] known-bug: #110338

use core::pin::Pin;
use std::future::Future;

pub trait Foo<'a> {
    type Future: Future<Output = ()>;

    fn foo() -> Self::Future;
}

struct MyType<T>(T);

impl<'a, T> Foo<'a> for MyType<T>
where
    T: Foo<'a>,
    T::Future: Send,
{
    type Future = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

    fn foo() -> Self::Future {
        Box::pin(async move { <T as Foo<'a>>::foo().await })
    }
}

fn main() {}
