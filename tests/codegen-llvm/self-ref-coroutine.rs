// Tests that the coroutine struct is not noalias if it has self-referential
// fields.

//@ compile-flags: -C opt-level=3
//@ edition: 2021

#![crate_type = "lib"]

use std::future::Future;
use std::pin::Pin;

async fn inner() {}

// CHECK-LABEL: ; self_ref_coroutine::my_async_fn::{closure#0}
// CHECK-LABEL: my_async_fn
// CHECK-NOT: noalias
// CHECK-SAME: %_1
async fn my_async_fn(b: bool) -> i32 {
    let x = Box::new(5);
    let y = &x;
    if b {
        inner().await;
        std::hint::black_box(y);
    }
    *x + 1
}

#[no_mangle]
pub fn create_future_as_trait(b: bool) -> Pin<Box<dyn Future<Output = i32>>> {
    Box::pin(my_async_fn(b))
}
