//@ edition: 2021
//@ check-pass

#![feature(async_trait_bounds)]

async fn foo() {}

async fn call_asyncly(f: impl async Fn(i32) -> i32) -> i32 {
    f(1).await
}

fn main() {
    let fut = call_asyncly(|x| async move { x + 1 });
}
