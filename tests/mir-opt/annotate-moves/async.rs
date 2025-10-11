//@ compile-flags: -Z annotate-moves=1 -C debuginfo=full
//@ ignore-std-debug-assertions
//@ edition: 2021

#![crate_type = "lib"]

#[derive(Clone)]
pub struct LargeStruct {
    pub data: [u64; 20], // 160 bytes
}

// EMIT_MIR async.test_async.AnnotateMoves.after.mir
pub async fn test_async(s: LargeStruct) -> LargeStruct {
    // CHECK-LABEL: fn test_async(
    // Async generates a state machine that moves values across await points
    // The move may show up when constructing the future state
    // CHECK: scope {{[0-9]+}} (inlined core::profiling::compiler_move::<LargeStruct, 160>)
    s
}

async fn make_future() -> LargeStruct {
    LargeStruct { data: [0; 20] }
}

async fn consume_future<F: std::future::Future>(f: F) -> F::Output {
    f.await
}

// EMIT_MIR async.test_future_move.AnnotateMoves.after.mir
pub async fn test_future_move() -> LargeStruct {
    // CHECK-LABEL: fn test_future_move(
    // Moving the future type itself (the state machine) when passing to consume_future
    // CHECK: scope {{[0-9]+}} (inlined core::profiling::compiler_move::<{async fn body of make_future()}
    let fut = make_future();
    consume_future(fut).await
}
