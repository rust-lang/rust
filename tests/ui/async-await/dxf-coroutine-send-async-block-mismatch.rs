//@ revisions: without_aob with_aob
//@ edition: 2024
//@ [without_aob] compile-flags: -Znext-solver -Zdxf
//@ [without_aob] known-bug: #126550
//@ [with_aob] compile-flags: -Znext-solver -Zdxf -Zassumptions-on-binders
//@ [with_aob] check-pass

// Minimized from a futures-util join_all/then/map pattern.
//
// The bug chain:
// 1. MIR erases free regions to `ReErased` as usual.
// 2. `coroutine_hidden_types` assigns each erased region a distinct `BoundVar`.
// 3. Auto-trait solver instantiates bound vars as placeholders.
// 4. Projection normalization on `MaybeDone::Done(...)` triggers a check on
//    the `F: FnOnce(Fut::Output)` where-clause through the Flatten<Map<...>>
//    chain, requiring `for<'a> FnOnce(&'a ())` but closure only implements
//    `FnOnce(&'static ())`.
//
// Unlike #126551, the reference is in the closure's `async move { &THING }`.
//
// With `-Zassumptions-on-binders`, the solver can use the NLL data `'a: 'static`
// to prove the higher-ranked bound.

#![allow(dead_code)]
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

const THING: () = ();

struct Map<Fut, F>(Fut, F);

impl<Fut: Future, F: FnOnce(Fut::Output) -> T, T> Future for Map<Fut, F> {
    type Output = T;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<T> { todo!() }
}

enum Flatten<Fut1, Fut2> {
    First(Fut1),
    Second(Fut2),
}

impl<Fut: Future> Future for Flatten<Fut, Fut::Output>
where
    Fut::Output: Future,
{
    type Output = <Fut::Output as Future>::Output;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> { todo!() }
}

enum MaybeDone<Fut: Future> {
    Future(Fut),
    Done(Fut::Output),
}

impl<Fut: Future> Future for MaybeDone<Fut> {
    type Output = ();
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> { todo!() }
}

async fn foo() {
    MaybeDone::Future(Flatten::First(Map(async {}, |()| async move { &THING }))).await;
}

fn trouble() -> impl Send {
    foo()
}

fn main() {
    trouble();
}
