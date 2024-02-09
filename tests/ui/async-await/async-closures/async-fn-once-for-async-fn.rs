// aux-build:block-on.rs
// edition:2021
// run-pass

// FIXME(async_closures): When `fn_sig_for_fn_abi` is fixed, remove this.
// ignore-pass (test emits codegen-time warnings)

#![feature(async_closure)]

extern crate block_on;

fn main() {
    block_on::block_on(async {
        let x = async || {};

        async fn needs_async_fn_once(x: impl async FnOnce()) {
            x().await;
        }
        needs_async_fn_once(x).await;
    });
}
