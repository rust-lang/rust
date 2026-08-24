//@compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes

// This is a regression test for a Miri ICE when encountering a `SyntheticCoroutineBody` from an external crate (see
// `cross_crate_items` for details). Calling the async closure triggered the ICE: https://github.com/rust-lang/rust/issues/156905

use std::future::Future;
use std::task::{Context, Poll, Waker};

use futures::SinkExt as _;

// Taken from tests/pass/async-fn.rs.
fn run_fut<T>(fut: impl Future<Output = T>) -> T {
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
    let mut sink = Box::pin(cross_crate_items::returns_async_closure());
    run_fut(sink.send(())).unwrap();
}
