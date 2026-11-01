//@ check-pass
//@ edition: 2021

// Make sure that we don't hit a query cycle when validating
// the by-move coroutine body for an async closure.

use std::future::Future;

async fn test<Fut: Future>(operation: impl Fn() -> Fut) {
    operation().await;
}

pub async fn orchestrate_simple_crud() {
    test(async || async {}.await).await;
}

fn main() {}
