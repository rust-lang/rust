// gate-test-await_macro
// edition:2018

#![feature(async_await)]

async fn bar() {}

async fn foo() {
    await!(bar()); //~ ERROR `await!(<expr>)` macro syntax is unstable, and will soon be removed
}

fn main() {}
