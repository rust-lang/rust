#![feature(noop_waker)]
use std::future::Future;

enum Runtime {}

async fn run(_: Runtime) {}

async fn apply_timeout() {
    let c = async {};
    match None::<Runtime> {
        None => {
            c.await;
        },
        Some(r) => {
            run(r).await;
        }
    }
}

fn main() {
    let waker = std::task::Waker::noop();
    let mut ctx = std::task::Context::from_waker(&waker);
    let fut = std::pin::pin!(apply_timeout());
    let _ = fut.poll(&mut ctx);
}
