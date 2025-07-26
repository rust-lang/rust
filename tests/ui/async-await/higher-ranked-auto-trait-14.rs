// Repro for <https://github.com/rust-lang/rust/issues/124757#issue-2279603232>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] check-pass
//@[no_assumptions] known-bug: #110338

use std::collections::HashSet;
use std::future::Future;

trait MyTrait {
    fn blah(&self, x: impl Iterator<Item = u32>) -> impl Future<Output = ()> + Send;
}

fn foo<T: MyTrait + Send + Sync>(
    val: T,
    unique_x: HashSet<u32>,
) -> impl Future<Output = ()> + Send {
    let cached = HashSet::new();
    async move {
        let xs = unique_x.union(&cached)
            // .copied() // works
            .map(|x| *x) // error
            ;
        let blah = val.blah(xs.into_iter()).await;
    }
}

fn main() {}
