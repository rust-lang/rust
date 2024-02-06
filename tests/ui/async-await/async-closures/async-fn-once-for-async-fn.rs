// aux-build:block-on.rs
// edition:2021
// run-pass

// FIXME(async_closures): When `fn_sig_for_fn_abi` is fixed, remove this.
// ignore-pass (test emits codegen-time warnings)

#![feature(async_closure, async_fn_traits)]

extern crate block_on;

use std::ops::AsyncFnOnce;

fn main() {
    block_on::block_on(async {
        let x = async || {};

        async fn needs_async_fn_once(x: impl AsyncFnOnce()) {
            x().await;
        }
        needs_async_fn_once(x).await;
    });
}
