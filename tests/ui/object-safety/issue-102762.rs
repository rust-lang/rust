//@ compile-flags: --crate-type=lib
// This test checks that the `where_clauses_object_safety` lint does not cause
// other object safety *hard errors* to be suppressed, because we currently
// only emit one object safety error per trait...

use std::future::Future;
use std::pin::Pin;

pub trait Fetcher: Send + Sync {
    fn get<'a>(self: &'a Box<Self>) -> Pin<Box<dyn Future<Output = Vec<u8>> + 'a>>
    where
        Self: Sync,
    {
        todo!()
    }
}

fn fetcher() -> Box<dyn Fetcher> {
    //~^ ERROR the trait `Fetcher` cannot be made into an object
    todo!()
}

pub fn foo() {
    let fetcher = fetcher();
    //~^ ERROR the trait `Fetcher` cannot be made into an object
    let _ = fetcher.get();
    //~^ ERROR the trait `Fetcher` cannot be made into an object
}
