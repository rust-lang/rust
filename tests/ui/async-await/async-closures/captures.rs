//@ aux-build:block-on.rs
//@ edition:2021
//@ run-pass
//@ check-run-results

// Same as miri's `tests/pass/async-closure-captures.rs`, keep in sync

extern crate block_on;

fn main() {
    block_on::block_on(async_main());
}

async fn call<T>(f: &impl AsyncFn() -> T) -> T {
    f().await
}

async fn call_once<T>(f: impl AsyncFnOnce() -> T) -> T {
    f().await
}

#[derive(Debug)]
#[allow(unused)]
struct Hello(i32);

async fn async_main() {
    // Capture something by-ref
    {
        let x = Hello(0);
        let c = async || {
            println!("{x:?}");
        };
        call(&c).await;
        call_once(c).await;

        let x = &Hello(1);
        let c = async || {
            println!("{x:?}");
        };
        call(&c).await;
        call_once(c).await;
    }

    // Capture something and consume it (force to `AsyncFnOnce`)
    {
        let x = Hello(2);
        let c = async || {
            println!("{x:?}");
            drop(x);
        };
        call_once(c).await;
    }

    // Capture something with `move`, don't consume it
    {
        let x = Hello(3);
        let c = async move || {
            println!("{x:?}");
        };
        call(&c).await;
        call_once(c).await;

        let x = &Hello(4);
        let c = async move || {
            println!("{x:?}");
        };
        call(&c).await;
        call_once(c).await;
    }

    // Capture something with `move`, also consume it (so `AsyncFnOnce`)
    {
        let x = Hello(5);
        let c = async move || {
            println!("{x:?}");
            drop(x);
        };
        call_once(c).await;
    }

    fn force_fnonce<T>(f: impl AsyncFnOnce() -> T) -> impl AsyncFnOnce() -> T {
        f
    }

    // Capture something with `move`, but infer to `AsyncFnOnce`
    {
        let x = Hello(6);
        let c = force_fnonce(async move || {
            println!("{x:?}");
        });
        call_once(c).await;

        let x = &Hello(7);
        let c = force_fnonce(async move || {
            println!("{x:?}");
        });
        call_once(c).await;
    }

    // Capture something by-ref, but infer to `AsyncFnOnce`
    {
        let x = Hello(8);
        let c = force_fnonce(async || {
            println!("{x:?}");
        });
        call_once(c).await;

        let x = &Hello(9);
        let c = force_fnonce(async || {
            println!("{x:?}");
        });
        call_once(c).await;
    }
}
