//@ run-pass
//@ edition: 2021
#![feature(async_iterator, async_iter_from_iter, async_for_loop)]

use std::future::Future;

// make sure a simple for await loop works
async fn real_main() {
    let iter = core::async_iter::from_iter(0..3);
    let mut count = 0;
    for await i in iter {
        assert_eq!(i, count);
        count += 1;
    }
    assert_eq!(count, 3);
}

fn main() {
    let future = real_main();
    let mut cx = &mut core::task::Context::from_waker(std::task::Waker::noop());
    let mut future = core::pin::pin!(future);
    while let core::task::Poll::Pending = future.as_mut().poll(&mut cx) {}
}
