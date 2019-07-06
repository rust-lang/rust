// run-pass

// compile-flags: --edition=2018

#![feature(async_await)]

pub enum Uninhabited { }

fn uninhabited_async() -> Uninhabited {
    unreachable!()
}

async fn noop() { }

#[allow(unused)]
async fn contains_never() {
    let error = uninhabited_async();
    noop().await;
    let error2 = error;
}

#[allow(unused_must_use)]
fn main() {
    contains_never();
}
