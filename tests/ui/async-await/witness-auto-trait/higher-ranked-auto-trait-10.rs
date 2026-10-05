// Repro for <https://github.com/rust-lang/rust/issues/92415#issue-1090723521>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] known-bug: unknown
//@[no_assumptions] known-bug: #110338

use std::any::Any;
use std::future::Future;

trait Foo<'a>: Sized {
    type Error;
    fn foo(x: &'a str) -> Result<Self, Self::Error>;
}

impl<'a> Foo<'a> for &'a str {
    type Error = ();

    fn foo(x: &'a str) -> Result<Self, Self::Error> {
        Ok(x)
    }
}

async fn get_foo<'a, T>(x: &'a str) -> Result<T, <T as Foo<'a>>::Error>
where
    T: Foo<'a>,
{
    Foo::foo(x)
}

fn bar<'a>(x: &'a str) -> Box<dyn Future<Output = Result<&'a str, ()>> + Send + 'a> {
    Box::new(async move { get_foo(x).await })
}

fn main() {}
