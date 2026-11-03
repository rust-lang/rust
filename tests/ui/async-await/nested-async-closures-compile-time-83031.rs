//! Regression test for <https://github.com/rust-lang/rust/issues/83031>.

//@ check-pass
//@ edition: 2018

use std::future::Future;

macro_rules! compose_middleware_inner {
    ($route:ident, $first:ident, $second:ident, $($tail:ident),+) => {
        $first(|| async {
            compose_middleware_inner!($route, $second, $($tail),+)
        }).await
    };
    ($route:ident, $first:ident, $second:ident) => {
        $first(|| async move { $second($route).await }).await
    };
}

macro_rules! compose_middleware {
    ($name:ident, $($tail:ident),+) => {
        pub async fn $name<N, Fut>(route: N)
        where
            N: FnOnce() -> Fut,
            Fut: Future<Output = ()>,
        {
            compose_middleware_inner!(route, $($tail),+)
        }
    };
}

async fn log<N, Fut>(next: N)
where
    N: FnOnce() -> Fut,
    Fut: Future<Output = ()>,
{
    next().await;
}

compose_middleware!(
    my_middleware,
    log, log, log, log, log, log, log, log, log, log,
    log, log, log, log, log, log, log, log, log, log
);

fn main() {}
