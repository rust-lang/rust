#![warn(clippy::unused_async)]
#![allow(incomplete_features)]

use std::future::Future;
use std::pin::Pin;

mod issue10800 {
    #![allow(dead_code, unused_must_use, clippy::no_effect)]

    use std::future::ready;

    async fn async_block_await() {
        //~^ unused_async

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

mod issue9695 {
    use std::future::Future;

    async fn f() {}
    async fn f2() {}
    async fn f3() {}
    //~^ unused_async

    fn needs_async_fn<F: Future<Output = ()>>(_: fn() -> F) {}

    fn test() {
        let x = f;
        needs_async_fn(x); // async needed in f
        needs_async_fn(f2); // async needed in f2
        f3(); // async not needed in f3
    }
}

mod issue13466 {
    use std::future::Future;

    struct Wrap<F>(F);
    impl<F> From<F> for Wrap<F> {
        fn from(f: F) -> Self {
            Self(f)
        }
    }
    fn takes_fut<F: Fn() -> Fut, Fut: Future>(_: Wrap<F>) {}
    async fn unused_async() {}
    fn fp() {
        takes_fut(unused_async.into());
    }
}

async fn foo() -> i32 {
    //~^ unused_async

    4
}

async fn bar() -> i32 {
    foo().await
}

struct S;

impl S {
    async fn unused(&self) -> i32 {
        //~^ unused_async

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

mod issue14704 {
    use std::sync::Arc;

    trait Action {
        async fn cancel(self: Arc<Self>) {}
    }
}
