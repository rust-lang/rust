//@ aux-build:block-on.rs
//@ edition:2021
//@ run-pass
//@ check-run-results

extern crate block_on;

async fn call_once(f: impl AsyncFnOnce()) {
    f().await;
}

async fn async_main() {
    let x = &mut 0;
    let y = &mut 0;
    let c = async || {
        *x = 1;
        *y = 2;
    };
    call_once(c).await;
    println!("{x} {y}");
}

fn main() {
    block_on::block_on(async_main());
}
