// edition:2018
//
// Tests that the .await syntax can't be used to make a generator

#![feature(async_await)]

async fn foo() {}

fn make_generator() {
    let _gen = || foo().await;
    //~^ ERROR `await` is only allowed inside `async` functions and blocks
}

fn main() {}
