// Regression test for #142559
//@ run-pass
//@ compile-flags: -Clink-dead-code=true
#![feature(async_drop)]
#![allow(incomplete_features)]

//@ edition: 2024

async fn run<F: Future>(f: impl Fn() -> F) {
    f().await;
}

pub async fn async_drop_async_closure() {
    let x = async || async {}.await;

    run(x).await;
}

fn main() {}
