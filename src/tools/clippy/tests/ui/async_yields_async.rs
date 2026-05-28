#![warn(clippy::async_yields_async)]
#![allow(clippy::redundant_async_block)]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

struct CustomFutureType;

impl Future for CustomFutureType {
    type Output = u8;

    fn poll(self: Pin<&mut Self>, _: &mut Context) -> Poll<Self::Output> {
        Poll::Ready(3)
    }
}

fn custom_future_type_ctor() -> CustomFutureType {
    CustomFutureType
}

async fn f() -> CustomFutureType {
    // Don't warn for functions since you have to explicitly declare their
    // return types.
    CustomFutureType
}

#[rustfmt::skip]
fn main() {
    let _f = {
        3
    };
    let _g = async {
        3
    };
    let _h = async {
        async {
            3
        }
    };
    //~^^^^ async_yields_async
    let _i = async {
        CustomFutureType
    };
    //~^^ async_yields_async
    let _i = async || {
        3
    };
    let _j = async || {
        async {
            3
        }
    };
    //~^^^^ async_yields_async
    let _k = async || {
        CustomFutureType
    };
    //~^^ async_yields_async
    let _l = async || CustomFutureType;
    //~^ async_yields_async
    let _m = async || {
        println!("I'm bored");
        // Some more stuff

        // Finally something to await
        CustomFutureType
    };
    //~^^ async_yields_async
    let _n = async || custom_future_type_ctor();
    let _o = async || f();
}

#[rustfmt::skip]
#[allow(dead_code)]
fn check_expect_suppression() {
    #[expect(clippy::async_yields_async)]
    let _j = async || {
        async {
            3
        }
    };
}

#[allow(clippy::let_underscore_future)]
fn issue15552() {
    async fn bar(i: i32) {}

    macro_rules! call_bar {
        () => {
            async { bar(5) }
        };
        ($e:expr) => {
            bar($e)
        };
    }
    let x = async { call_bar!(5) };
    //~^ async_yields_async
    let y = async { call_bar!() };
    //~^ async_yields_async
    //~| async_yields_async

    use std::future::{Future, Ready};
    use std::ops::Add;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    struct CustomFutureType;
    impl Add for CustomFutureType {
        type Output = Self;
        fn add(self, other: Self) -> Self {
            self
        }
    }
    impl Future for CustomFutureType {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Ready(())
        }
    }
    let _ = async { CustomFutureType + CustomFutureType };
    //~^ async_yields_async
}
