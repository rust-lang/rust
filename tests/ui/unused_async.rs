#![warn(clippy::unused_async)]
#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

use std::future::Future;
use std::pin::Pin;

mod issue10800 {
    #![allow(dead_code, unused_must_use, clippy::no_effect)]

    use std::future::ready;

    async fn async_block_await() {
        async {
            ready(()).await;
        };
    }

    async fn normal_block_await() {
        {
            {
                ready(()).await;
            }
        }
    }
}

mod issue10459 {
    trait HasAsyncMethod {
        async fn do_something() -> u32;
    }

    impl HasAsyncMethod for () {
        async fn do_something() -> u32 {
            1
        }
    }
}

async fn foo() -> i32 {
    4
}

async fn bar() -> i32 {
    foo().await
}

struct S;

impl S {
    async fn unused(&self) -> i32 {
        1
    }

    async fn used(&self) -> i32 {
        self.unused().await
    }
}

trait AsyncTrait {
    fn trait_method() -> Pin<Box<dyn Future<Output = i32>>>;
}

macro_rules! async_trait_impl {
    () => {
        impl AsyncTrait for S {
            fn trait_method() -> Pin<Box<dyn Future<Output = i32>>> {
                async fn unused() -> i32 {
                    5
                }

                Box::pin(unused())
            }
        }
    };
}
async_trait_impl!();

fn main() {
    foo();
    bar();
}
