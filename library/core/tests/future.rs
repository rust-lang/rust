use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Wake};
use std::thread;

fn block_on<F: Future>(fut: F) -> F::Output {
    struct Waker;
    impl Wake for Waker {
        fn wake(self: Arc<Self>) {
            thread::current().unpark()
        }
    }

    let waker = Arc::new(Waker).into();
    let mut cx = Context::from_waker(&waker);
    let mut fut = Box::pin(fut);

    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(value) => break value,
            Poll::Pending => thread::park(),
        }
    }
}

#[test]
fn test_map() {
    let future = async { 1 };
    let future = future.map(|x| x + 3);
    assert_eq!(block_on(future), 4);
}
