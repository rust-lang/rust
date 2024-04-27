//@ check-pass
//@ edition: 2021

#![feature(async_closure)]

async fn foo(x: impl async Fn(&str) -> &str) {}

fn main() {
    foo(async |x| x);
}
