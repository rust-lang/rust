//@ run-pass
//@ edition: 2024
#![feature(async_iterator, async_iter_from_iter, async_for_loop, gen_blocks)]

async gen fn async_iter() -> i32 {
    let iter = core::async_iter::from_iter(0..3);
    for await i in iter {
        yield i + 1;
    }
}

// make sure a simple for await loop works
async fn real_main() {
    let mut count = 1;
    for await i in async_iter() {
        assert_eq!(i, count);
        count += 1;
    }
    assert_eq!(count, 4);
}

fn main() {
    let future = real_main();
    let mut cx = &mut core::task::Context::from_waker(std::task::Waker::noop());
    let mut future = core::pin::pin!(future);
    while let core::task::Poll::Pending = future.as_mut().poll(&mut cx) {}
}
