#![warn(clippy::unused_async)]

use std::future::Future;
use std::pin::Pin;

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
