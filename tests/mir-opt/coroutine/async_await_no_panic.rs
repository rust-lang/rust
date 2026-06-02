// This test is the same as `async_await`, except it disables panics in the statemachines.
// Instead of emitting false assertions, the basic blocks here should return
// `CoroutineState::Yielded(const ())`

//@ edition:2018
//@ compile-flags: -Zmir-opt-level=0 -Zasync-panic=false
//@ needs-unwind

#![crate_type = "lib"]

// EMIT_MIR async_await_no_panic.a-{closure#0}.StateTransform.diff
async fn a() {
    // CHECK-LABEL: fn a::{closure#0}(
    // CHECK-SAME: _1: Pin<&mut {static coroutine body of a()}>
    // CHECK-SAME: _2: std::future::ResumeTy
    // CHECK-SAME: -> CoroutineState<(), ()>
    // CHECK-NOT: get_context
    // CHECK-NOT: assert
    // CHECK: CoroutineState::<(), ()>::Yielded
    // CHECK-NOT: get_context
    // CHECK-NOT: assert
}

// EMIT_MIR async_await_no_panic.b-{closure#0}.StateTransform.diff
pub async fn b() {
    // CHECK-LABEL: fn b::{closure#0}(
    // CHECK-SAME: _1: Pin<&mut {static coroutine body of b()}>
    // CHECK-SAME: _2: std::future::ResumeTy
    // CHECK-SAME: -> CoroutineState<(), ()>
    // CHECK-NOT: get_context
    // CHECK-NOT: assert
    // CHECK: CoroutineState::<(), ()>::Yielded
    // CHECK-NOT: get_context
    // CHECK-NOT: assert
    a().await;
    a().await
}

// CHECK-NOT: panic
