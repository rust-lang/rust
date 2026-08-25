// regression test for <https://github.com/rust-lang/rust/issues/101650>
// assert that Future which has format!() with an async function is Send

#![allow(unused)]

//@ check-pass
//@ edition: 2018

use core::future::Future;
use core::pin::Pin;

fn build_string() -> Pin<Box<dyn Future<Output = String> + Send>> {
    Box::pin(async move {
        let mut string_builder = String::new();
        string_builder += &format!("Hello {}", helper().await);
        string_builder
    })
}

async fn helper() -> String {
    "World".to_string()
}

fn main() {}
