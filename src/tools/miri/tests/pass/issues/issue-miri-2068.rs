use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use std::sync::Arc;

struct NopWaker;

impl std::task::Wake for NopWaker {
    fn wake(self: Arc<Self>) {}
}

pub fn fuzzing_block_on<O, F: Future<Output = O>>(fut: F) -> O {
    let mut fut = std::pin::pin!(fut);
    let waker = std::task::Waker::from(Arc::new(NopWaker));
    let mut context = std::task::Context::from_waker(&waker);
    loop {
        match fut.as_mut().poll(&mut context) {
            Poll::Ready(v) => return v,
            Poll::Pending => {}
        }
    }
}

pub struct LastFuture<S> {
    last: S,
}

impl<S> Future for LastFuture<S>
where
    Self: Unpin,
    S: Unpin + Copy,
{
    type Output = S;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        return Poll::Ready(self.last);
    }
}

fn main() {
    fuzzing_block_on(async {
        LastFuture { last: &0u32 }.await;
        LastFuture { last: Option::<u32>::None }.await;
    });
}
