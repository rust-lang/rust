//@ skip-filecheck
//@ compile-flags: -Zmir-opt-level=0
//@ needs-unwind
//@ edition: 2024

#![feature(never_type)]

use std::future::Future;

// See if we can run a basic `async fn`
// EMIT_MIR async_fn.foo-{closure#0}.StateTransform.diff
// EMIT_MIR async_fn.foo-{closure#0}.coroutine_drop.0.mir
// EMIT_MIR async_fn.foo-{closure#0}.coroutine_drop_proxy_async.0.mir
// EMIT_MIR async_fn.foo-{closure#0}-{closure#0}.StateTransform.diff
// EMIT_MIR async_fn.foo-{closure#0}-{closure#0}.coroutine_drop.0.mir
// EMIT_MIR async_fn.foo-{closure#0}-{closure#0}.coroutine_drop_proxy_async.0.mir
pub async fn foo(x: &u32, y: u32) -> u32 {
    let y = &y;
    let z = 9;
    let z = &z;
    let y = async { *y + *z }.await;
    let a = 10;
    let a = &a;
    *x + y + *a
}

// EMIT_MIR async_fn.add-{closure#0}.StateTransform.diff
// EMIT_MIR async_fn.add-{closure#0}.coroutine_drop.0.mir
// EMIT_MIR async_fn.add-{closure#0}.coroutine_drop_proxy_async.0.mir
// EMIT_MIR async_fn.add-{closure#0}-{closure#0}.StateTransform.diff
// EMIT_MIR async_fn.add-{closure#0}-{closure#0}.coroutine_drop.0.mir
// EMIT_MIR async_fn.add-{closure#0}-{closure#0}.coroutine_drop_proxy_async.0.mir
async fn add(x: u32, y: u32) -> u32 {
    let a = async { x + y };
    a.await
}

// EMIT_MIR async_fn.build_aggregate-{closure#0}.StateTransform.diff
// EMIT_MIR async_fn.build_aggregate-{closure#0}.coroutine_drop.0.mir
// EMIT_MIR async_fn.build_aggregate-{closure#0}.coroutine_drop_proxy_async.0.mir
async fn build_aggregate(a: u32, b: u32, c: u32, d: u32) -> u32 {
    let x = (add(a, b).await, add(c, d).await);
    x.0 + x.1
}

enum Never {}
fn never() -> Never {
    panic!()
}

// EMIT_MIR async_fn.includes_never-{closure#0}.StateTransform.diff
// EMIT_MIR async_fn.includes_never-{closure#0}.coroutine_drop.0.mir
// EMIT_MIR async_fn.includes_never-{closure#0}.coroutine_drop_proxy_async.0.mir
// EMIT_MIR async_fn.includes_never-{closure#0}-{closure#0}.StateTransform.diff
// EMIT_MIR async_fn.includes_never-{closure#0}-{closure#0}.coroutine_drop.0.mir
// EMIT_MIR async_fn.includes_never-{closure#0}-{closure#0}.coroutine_drop_proxy_async.0.mir
// EMIT_MIR async_fn.includes_never-{closure#0}-{closure#1}.StateTransform.diff
// EMIT_MIR async_fn.includes_never-{closure#0}-{closure#1}.coroutine_drop.0.mir
// EMIT_MIR async_fn.includes_never-{closure#0}-{closure#1}.coroutine_drop_proxy_async.0.mir
async fn includes_never(crash: bool, x: u32) -> u32 {
    let result = async { x * x }.await;
    if !crash {
        return result;
    }
    #[allow(unused)]
    let bad = never();
    result *= async { x + x }.await;
    drop(bad);
    result
}

// EMIT_MIR async_fn.partial_init-{closure#0}.StateTransform.diff
// EMIT_MIR async_fn.partial_init-{closure#0}.coroutine_drop.0.mir
// EMIT_MIR async_fn.partial_init-{closure#0}.coroutine_drop_proxy_async.0.mir
// EMIT_MIR async_fn.partial_init-{closure#0}-{closure#0}.StateTransform.diff
// EMIT_MIR async_fn.partial_init-{closure#0}-{closure#0}.coroutine_drop.0.mir
// EMIT_MIR async_fn.partial_init-{closure#0}-{closure#0}.coroutine_drop_proxy_async.0.mir
async fn partial_init(x: u32) -> u32 {
    #[allow(unreachable_code)]
    let _x: (String, !) = (String::new(), return async { x + x }.await);
}

async fn read_exact(_from: &mut &[u8], _to: &mut [u8]) -> Option<()> {
    Some(())
}

// EMIT_MIR async_fn.hello_world-{closure#0}.StateTransform.diff
// EMIT_MIR async_fn.hello_world-{closure#0}.coroutine_drop.0.mir
// EMIT_MIR async_fn.hello_world-{closure#0}.coroutine_drop_proxy_async.0.mir
async fn hello_world() {
    let data = [0u8; 1];
    let mut reader = &data[..];

    let mut marker = [0u8; 1];
    read_exact(&mut reader, &mut marker).await.unwrap();
}

// This example comes from https://github.com/rust-lang/rust/issues/115145
// EMIT_MIR async_fn.uninhabited_variant-{closure#0}.StateTransform.diff
// EMIT_MIR async_fn.uninhabited_variant-{closure#0}.coroutine_drop.0.mir
// EMIT_MIR async_fn.uninhabited_variant-{closure#0}.coroutine_drop_proxy_async.0.mir
#[allow(unreachable_patterns)]
async fn uninhabited_variant() {
    async fn unreachable(_: Never) {}

    let c = async {};
    match None::<Never> {
        None => {
            c.await;
        }
        Some(r) => {
            unreachable(r).await;
        }
    }
}

fn run_fut<T>(fut: impl Future<Output = T>) -> T {
    use std::task::{Context, Poll, Waker};

    let mut context = Context::from_waker(Waker::noop());

    let mut pinned = Box::pin(fut);
    loop {
        match pinned.as_mut().poll(&mut context) {
            Poll::Pending => continue,
            Poll::Ready(v) => return v,
        }
    }
}

fn main() {
    let x = 5;
    assert_eq!(run_fut(foo(&x, 7)), 31);
    assert_eq!(run_fut(build_aggregate(1, 2, 3, 4)), 10);
    assert_eq!(run_fut(includes_never(false, 4)), 16);
    assert_eq!(run_fut(partial_init(4)), 8);
    run_fut(hello_world());
    run_fut(uninhabited_variant());
}
