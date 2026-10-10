//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ edition: 2024
//@ check-pass

// Regression test for the `typesensei` crater breakage caused by
// trait-system-refactor-initiative#159. Getting an incorrect
// `batch_action::{opaque}: 'a` implied bound means there are now
// two ways to prove that `Action<'a, batch_action::{opaque}>` is
// well-formed. This causes us to emit a type test instead of a
// region constraint, causing this to fail as type tests are checked
// on the frozen region graph.

use std::{future::Future, marker::PhantomData};

pub fn batch_emplace<'a>(s: &'a str) -> Action<'a, impl Future + 'a> {
    if false {
        let n: Action<'a, _> = loop {};
        n
    } else {
        new(s, batch_action(s))
    }
}

// The outlive bound is necessary.
pub struct Action<'a, Fut: 'a> {
    _phantom: PhantomData<(&'a str, Fut)>,
}
fn new<'a, Fut>(api: &'a str, fut: Fut) -> Action<'a, Fut> {
    loop {}
}

fn batch_action<'a>(s: &'a str) -> impl Future + 'a {
    async {}
}

fn main() {}
