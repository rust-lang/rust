//@ edition:2018
//@compile-flags: --diagnostic-width=300
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::future::Future;
use std::ops::Coroutine;

async fn async_fn() {}
fn returns_async_block() -> impl Future<Output = ()> {
    async {}
}
fn returns_coroutine() -> impl Coroutine<(), Yield = (), Return = ()> {
    #[coroutine]
    || {
        let _: () = yield ();
    }
}

fn takes_future(_f: impl Future<Output = ()>) {}
fn takes_coroutine<ResumeTy>(_g: impl Coroutine<ResumeTy, Yield = (), Return = ()>) {}

fn main() {
    // okay:
    takes_future(async_fn());
    takes_future(returns_async_block());
    takes_future(async {});
    takes_coroutine(returns_coroutine());
    takes_coroutine(
        #[coroutine]
        || {
            let _: () = yield ();
        },
    );

    // async futures are not coroutines:
    takes_coroutine(async_fn());
    //~^ ERROR the trait bound
    takes_coroutine(returns_async_block());
    //~^ ERROR the trait bound
    takes_coroutine(async {});
    //~^ ERROR the trait bound

    // coroutines are not futures:
    takes_future(returns_coroutine());
    //~^ ERROR is not a future
    takes_future(
        #[coroutine]
        |ctx| {
            //~^ ERROR is not a future
            ctx = yield ();
        },
    );
}
