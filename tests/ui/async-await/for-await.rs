// run-pass
// edition: 2021
#![feature(async_stream, async_stream_from_iter, const_waker, async_for_loop, noop_waker)]

use std::future::Future;

// make sure a simple for await loop works
async fn real_main() {
    let iter = core::stream::from_iter(0..3);
    let mut count = 0;
    for await i in iter {
        assert_eq!(i, count);
        count += 1;
    }
    assert_eq!(count, 3);
}

fn main() {
    let future = real_main();
    let waker = std::task::Waker::noop();
    let mut cx = &mut core::task::Context::from_waker(&waker);
    let mut future = core::pin::pin!(future);
    while let core::task::Poll::Pending = future.as_mut().poll(&mut cx) {}
}
