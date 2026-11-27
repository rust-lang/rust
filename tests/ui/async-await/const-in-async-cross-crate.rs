//@ edition: 2024
//@ build-pass
//@ aux-build: const-in-async.rs

#![crate_type = "lib"]
#![deny(long_running_const_eval)]

extern crate const_in_async;

pub async fn call() {
    const_in_async::has_const().await;
}
