//@ edition: 2024
#![feature(async_drop)]
use std::future::AsyncDrop;
use std::pin::Pin;

pub async fn run() {
    let _st = St;
}

struct St;

impl Drop for St {
    fn drop(&mut self) {}
}

impl AsyncDrop for St {
    async fn drop(self: Pin<&mut Self>) {
        // Removing this line makes the program panic "normally" (not abort).
        nothing().await;
    }
}

async fn nothing() {}
