//@ aux-build:block-on.rs
//@ edition:2021
//@ run-pass

extern crate block_on;

fn main() {
    block_on::block_on(async {
        async fn needs_async_fn_once(x: impl AsyncFnOnce()) {
            x().await;
        }

        needs_async_fn_once(async || {}).await;

        needs_async_fn_once(|| async {}).await;

        async fn foo() {}
        needs_async_fn_once(foo).await;
    });
}
