use std::future::{join, Future};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake};
use std::thread;

struct PollN {
    val: usize,
    polled: usize,
    num: usize,
}

impl Future for PollN {
    type Output = usize;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.polled += 1;

        if self.polled == self.num {
            return Poll::Ready(self.val);
        }

        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

fn poll_n(val: usize, num: usize) -> PollN {
    PollN { val, num, polled: 0 }
}

#[test]
fn test_join() {
    block_on(async move {
        let x = join!(async { 0 });
        assert_eq!(x, 0);

        let x = join!(async { 0 }, async { 1 });
        assert_eq!(x, (0, 1));

        let x = join!(async { 0 }, async { 1 }, async { 2 });
        assert_eq!(x, (0, 1, 2));

        let x = join!(
            poll_n(0, 1),
            poll_n(1, 5),
            poll_n(2, 2),
            poll_n(3, 1),
            poll_n(4, 2),
            poll_n(5, 3),
            poll_n(6, 4),
            poll_n(7, 1)
        );
        assert_eq!(x, (0, 1, 2, 3, 4, 5, 6, 7));
    });
}

fn block_on(fut: impl Future) {
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
            Poll::Ready(_) => break,
            Poll::Pending => thread::park(),
        }
    }
}
