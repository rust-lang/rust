//@ aux-build:block-on.rs
//@ edition:2021
//@ run-pass
//@ check-run-results

extern crate block_on;

async fn for_each(f: impl AsyncFnOnce(&str) + Clone) {
    f.clone()("world").await;
    f.clone()("world2").await;
}

fn main() {
    block_on::block_on(async_main());
}

async fn async_main() {
    let x = String::from("Hello,");
    for_each(async move |s| {
        println!("{x} {s}");
    }).await;
}
