#![feature(async_closure)]
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
    let _i = async {
        CustomFutureType
    };
    let _i = async || {
        3
    };
    let _j = async || {
        async {
            3
        }
    };
    let _k = async || {
        CustomFutureType
    };
    let _l = async || CustomFutureType;
    let _m = async || {
        println!("I'm bored");
        // Some more stuff

        // Finally something to await
        CustomFutureType
    };
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
