// skip-filecheck
// This test makes sure that the coroutine MIR pass eliminates all calls to
// `get_context`, and that the MIR argument type for an async fn and all locals
// related to `yield` are `&mut Context`, and its return type is `Poll`.

//@ edition:2018
//@ compile-flags: -Zmir-opt-level=0 -C panic=abort

#![crate_type = "lib"]

// EMIT_MIR async_await.a-{closure#0}.coroutine_resume.0.mir
async fn a() {}

// EMIT_MIR async_await.b-{closure#0}.coroutine_resume.0.mir
pub async fn b() {
    a().await;
    a().await
}
