//@ edition:2024
// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(async_drop)]
#![allow(incomplete_features)]

// EMIT_MIR async_drop_live_dead.a-{closure#0}.coroutine_drop_async.0.mir
async fn a<T>(x: T) {}

fn main() {}
