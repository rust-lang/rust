//@ edition: 2024
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

use std::future::Future;

fn for_each<F, Fut>(_: F)
where
    F: FnMut(()) -> Fut,
    Fut: Future<Output = Result<(), ()>>,
{
}

fn main() {
    for_each(async |_| Ok(String::new()));
    //~^ ERROR mismatched types
}
