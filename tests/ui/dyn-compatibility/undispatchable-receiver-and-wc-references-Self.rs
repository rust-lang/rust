//@ compile-flags: --crate-type=lib
// This test checks that the `where_clauses_object_safety` lint does not cause
// other dyn-compatibility *hard errors* to be suppressed, because we currently
// only emit one dyn-compatibility error per trait...
// issue: rust-lang/rust#102762

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
    //~^ ERROR the trait `Fetcher` is not dyn compatible
    todo!()
}

pub fn foo() {
    let fetcher = fetcher();
    //~^ ERROR the trait `Fetcher` is not dyn compatible
    let _ = fetcher.get();
}
