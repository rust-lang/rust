// This test is the same as `async_await`, except it disables panics in the statemachines.
// Instead of emitting false assertions, the basic blocks here should return Poll::Pending

//@ edition:2018
//@ compile-flags: -Zmir-opt-level=0 -Zasync-panic=false
//@ needs-unwind

#![crate_type = "lib"]

// EMIT_MIR async_await_no_panic.a-{closure#0}.StateTransform.diff
async fn a() {
    // CHECK-LABEL: fn a::{closure#0}(
    // CHECK-SAME: _1: Pin<&mut {async fn body of a()}>
    // CHECK-SAME: _2: &mut Context<'_>
    // CHECK-SAME: -> Poll<()>
    // CHECK-NOT: get_context
}

// EMIT_MIR async_await_no_panic.b-{closure#0}.StateTransform.diff
pub async fn b() {
    // CHECK-LABEL: fn b::{closure#0}(
    // CHECK-SAME: _1: Pin<&mut {async fn body of b()}>
    // CHECK-SAME: _2: &mut Context<'_>
    // CHECK-SAME: -> Poll<()>
    // CHECK-NOT: get_context
    a().await;
    a().await
}

// CHECK-NOT: panic
