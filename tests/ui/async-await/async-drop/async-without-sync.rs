//@ edition: 2024
#![feature(async_drop)]
#![allow(incomplete_features)]
#![crate_type = "lib"]

use std::future::AsyncDrop;
use std::pin::Pin;

async fn foo() {
    let _st = St;
}

struct St;

impl AsyncDrop for St { //~ ERROR: `AsyncDrop` impl without `Drop` impl
    async fn drop(self: Pin<&mut Self>) {
        println!("123");
    }
}
