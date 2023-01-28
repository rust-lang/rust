// edition:2018
#![feature(generators, generator_trait)]

use std::future::Future;
use std::ops::Generator;

async fn async_fn() {}
fn returns_async_block() -> impl Future<Output = ()> {
    async {}
}
fn returns_generator() -> impl Generator<(), Yield = (), Return = ()> {
    || {
        let _: () = yield ();
    }
}

fn takes_future(_f: impl Future<Output = ()>) {}
fn takes_generator<ResumeTy>(_g: impl Generator<ResumeTy, Yield = (), Return = ()>) {}

fn main() {
    // okay:
    takes_future(async_fn());
    takes_future(returns_async_block());
    takes_future(async {});
    takes_generator(returns_generator());
    takes_generator(|| {
        let _: () = yield ();
    });

    // async futures are not generators:
    takes_generator(async_fn());
    //~^ ERROR the trait bound
    takes_generator(returns_async_block());
    //~^ ERROR the trait bound
    takes_generator(async {});
    //~^ ERROR the trait bound

    // generators are not futures:
    takes_future(returns_generator());
    //~^ ERROR is not a future
    takes_future(|ctx| {
        //~^ ERROR is not a future
        ctx = yield ();
    });
}
