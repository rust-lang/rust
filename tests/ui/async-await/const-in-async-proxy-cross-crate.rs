//@ build-pass
//@ edition: 2024
//@ aux-build: const_in_async_proxy.rs

#![feature(async_drop)]
#![allow(incomplete_features)]

#![crate_type = "lib"]
#![deny(long_running_const_eval)]

extern crate const_in_async_proxy;

pub async fn drop_future() {
    let mut future = const_in_async_proxy::has_const();

    // SAFETY: The future remains valid until its async destruction completes.
    unsafe {
        core::future::async_drop_in_place(&mut future).await;
    }
}
