// Regression test for <https://github.com/rust-lang/rust/issues/95719>.
// The `Send` bound of a GAT `impl Future` was lost when the future was
// wrapped in another `async fn`, failing with "the parameter type `G` may
// not live long enough".
//@ check-pass
//@ edition: 2021

#![feature(impl_trait_in_assoc_type)]

use std::future::Future;

pub trait Get: Send + Sync {
    type Ret<'a>: Future<Output = usize> + Send + 'a
    where
        Self: 'a;
    fn get<'a>(&'a self) -> Self::Ret<'a>
    where
        Self: 'a;
}

impl Get for usize {
    type Ret<'a>
        = impl Future<Output = usize> + Send + 'a
    where
        Self: 'a;

    fn get<'a>(&'a self) -> Self::Ret<'a>
    where
        Self: 'a,
    {
        async move { *self }
    }
}

fn is_send<R, F: Future<Output = R> + Send>(_f: &F) -> bool {
    true
}

async fn wrap<G: Get>(g: &G) -> usize {
    let fut = g.get();
    assert!(is_send(&fut));
    fut.await
}

async fn wrap_wrap<G: Get>(g: &G) -> usize {
    let fut = wrap(g);
    assert!(is_send(&fut));
    fut.await
}

fn main() {
    let _ = wrap_wrap(&0usize);
}
