#![feature(coverage_attribute)]
#![coverage(off)]
//@ edition: 2021

//@ aux-build: executor.rs
extern crate executor;

async fn ready() -> u8 {
    1
}

#[coverage(on)]
#[rustfmt::skip]
async fn await_ready() -> u8 {
    // await should be covered even if the function never yields
    ready()
        .await
}

fn main() {
    let mut future = Box::pin(await_ready());
    executor::block_on(future.as_mut());
}
