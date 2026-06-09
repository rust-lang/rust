#![allow(unused, clippy::manual_async_fn)]
#![warn(clippy::redundant_async_block)]

use std::future::{Future, IntoFuture};

async fn func1(n: usize) -> usize {
    n + 1
}

async fn func2() -> String {
    let s = String::from("some string");
    let f = async { (*s).to_owned() };
    let x = async { f.await };
    //~^ redundant_async_block
    x.await
}

fn main() {
    let fut1 = async { 17 };
    // Lint
    let fut2 = async { fut1.await };
    //~^ redundant_async_block

    let fut1 = async { 25 };
    // Lint
    let fut2 = async move { fut1.await };
    //~^ redundant_async_block

    // Lint
    let fut = async { async { 42 }.await };
    //~^ redundant_async_block

    // Do not lint: not a single expression
    let fut = async {
        func1(10).await;
        func2().await
    };

    // Do not lint: expression contains `.await`
    let fut = async { func1(func2().await.len()).await };
}

#[allow(clippy::let_and_return)]
fn capture_local() -> impl Future<Output = i32> {
    let fut = async { 17 };
    // Lint
    async move { fut.await }
    //~^ redundant_async_block
}

fn capture_local_closure(s: &str) -> impl Future<Output = &str> {
    let f = move || std::future::ready(s);
    // Do not lint: `f` would not live long enough
    async move { f().await }
}

#[allow(clippy::let_and_return)]
fn capture_arg(s: &str) -> impl Future<Output = &str> {
    let fut = async move { s };
    // Lint
    async move { fut.await }
    //~^ redundant_async_block
}

fn capture_future_arg<T>(f: impl Future<Output = T>) -> impl Future<Output = T> {
    // Lint
    async { f.await }
    //~^ redundant_async_block
}

fn capture_func_result<FN, F, T>(f: FN) -> impl Future<Output = T>
where
    F: Future<Output = T>,
    FN: FnOnce() -> F,
{
    // Do not lint, as f() would be evaluated prematurely
    async { f().await }
}

fn double_future(f: impl Future<Output = impl Future<Output = u32>>) -> impl Future<Output = u32> {
    // Do not lint, we will get a `.await` outside a `.async`
    async { f.await.await }
}

fn await_in_async<F, R>(f: F) -> impl Future<Output = u32>
where
    F: FnOnce() -> R,
    R: Future<Output = u32>,
{
    // Lint
    async { async { f().await + 1 }.await }
    //~^ redundant_async_block
}

#[derive(Debug, Clone)]
struct F {}

impl F {
    async fn run(&self) {}
}

pub async fn run() {
    let f = F {};
    let c = f.clone();
    // Do not lint: `c` would not live long enough
    spawn(async move { c.run().await });
    let _f = f;
}

fn spawn<F: Future + 'static>(_: F) {}

async fn work(_: &str) {}

fn capture() {
    let val = "Hello World".to_owned();
    // Do not lint: `val` would not live long enough
    spawn(async { work(&{ val }).await });
}

fn await_from_macro() -> impl Future<Output = u32> {
    macro_rules! mac {
        ($e:expr) => {
            $e.await
        };
    }
    // Do not lint: the macro may change in the future
    // or return different things depending on its argument
    async { mac!(async { 42 }) }
}

fn async_expr_from_macro() -> impl Future<Output = u32> {
    macro_rules! mac {
        () => {
            async { 42 }
        };
    }
    // Do not lint: the macro may change in the future
    async { mac!().await }
}

fn async_expr_from_macro_deep() -> impl Future<Output = u32> {
    macro_rules! mac {
        () => {
            async { 42 }
        };
    }
    // Do not lint: the macro may change in the future
    async { ({ mac!() }).await }
}

fn all_from_macro() -> impl Future<Output = u32> {
    macro_rules! mac {
        () => {
            // Lint
            async { async { 42 }.await }
            //~^ redundant_async_block
        };
    }
    mac!()
}

fn parts_from_macro() -> impl Future<Output = u32> {
    macro_rules! mac {
        ($e: expr) => {
            // Do not lint: `$e` might not always be side-effect free
            async { $e.await }
        };
    }
    mac!(async { 42 })
}

fn safe_parts_from_macro() -> impl Future<Output = u32> {
    macro_rules! mac {
        ($e: expr) => {
            // Lint
            async { async { $e }.await }
            //~^ redundant_async_block
        };
    }
    mac!(42)
}

fn parts_from_macro_deep() -> impl Future<Output = u32> {
    macro_rules! mac {
        ($e: expr) => {
            // Do not lint: `$e` might not always be side-effect free
            async { ($e,).0.await }
        };
    }
    let f = std::future::ready(42);
    mac!(f)
}

fn await_from_macro_deep() -> impl Future<Output = u32> {
    macro_rules! mac {
        ($e:expr) => {{ $e }.await};
    }
    // Do not lint: the macro may change in the future
    // or return different things depending on its argument
    async { mac!(async { 42 }) }
}

// Issue 11959
fn from_into_future(a: impl IntoFuture<Output = u32>) -> impl Future<Output = u32> {
    // Do not lint: `a` is not equivalent to this expression
    async { a.await }
}
