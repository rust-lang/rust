// Repro for <https://github.com/rust-lang/rust/issues/79648#issuecomment-749127947>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions assumptions_next
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions_next] compile-flags: -Zhigher-ranked-assumptions -Znext-solver
//@[assumptions] check-pass
//@[assumptions_next] check-pass
//@[no_assumptions] known-bug: #110338

// We're using this test to just check the way -Zhigher-ranked-assumptions
// interacts with the next-generation trait solver.
//[assumptions_next]~? WARN `-Zhigher-ranked-assumptions` is not supported with the next-generation trait solver

use std::future::Future;
use std::marker::PhantomData;

trait Stream {
    type Item;
}

struct Filter<St: Stream> {
    pending_item: St::Item,
}

fn filter<St: Stream>(_: St) -> Filter<St> {
    unimplemented!();
}

struct FilterMap<Fut, F> {
    f: F,
    pending: PhantomData<Fut>,
}

impl<Fut, F> Stream for FilterMap<Fut, F>
where
    F: FnMut() -> Fut,
    Fut: Future,
{
    type Item = ();
}

pub fn get_foo() -> impl Future + Send {
    async {
        let _y = &();
        let _x = filter(FilterMap {
            f: || async move { *_y },
            pending: PhantomData,
        });
        async {}.await;
        drop(_x);
    }
}

fn main() {}
