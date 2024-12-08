//@ build-pass
//@ edition: 2021

#![feature(async_closure)]

async fn foo(x: impl AsyncFn(&str) -> &str) {}

fn main() {
    foo(async |x| x);
}
