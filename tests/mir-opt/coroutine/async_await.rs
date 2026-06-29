// This test makes sure that the coroutine MIR pass eliminates all calls to
// `get_context`, and that the MIR argument type for an async fn and all locals
// related to `yield` are `&mut Context`, and its return type is `Poll`.

//@ edition:2018
//@ compile-flags: -Zmir-opt-level=0
//@ needs-unwind

#![crate_type = "lib"]

// EMIT_MIR async_await.a-{closure#0}.StateTransform.diff
async fn a() {
    // CHECK-LABEL: fn a::{closure#0}(
    // CHECK-SAME: _1: Pin<&mut {async fn body of a()}>
    // CHECK-SAME: _2: &mut Context<'_>
    // CHECK-SAME: -> Poll<()>
    // CHECK-NOT: get_context
}

// EMIT_MIR async_await.b-{closure#0}.StateTransform.diff
pub async fn b() {
    // CHECK-LABEL: fn b::{closure#0}(
    // CHECK-SAME: _1: Pin<&mut {async fn body of b()}>
    // CHECK-SAME: _2: &mut Context<'_>
    // CHECK-SAME: -> Poll<()>
    // CHECK-NOT: get_context
    a().await;
    a().await
}
