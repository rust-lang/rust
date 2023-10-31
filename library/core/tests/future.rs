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
        let x = join!(async { 0 }).await;
        assert_eq!(x, 0);

        let x = join!(async { 0 }, async { 1 }).await;
        assert_eq!(x, (0, 1));

        let x = join!(async { 0 }, async { 1 }, async { 2 }).await;
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
        )
        .await;
        assert_eq!(x, (0, 1, 2, 3, 4, 5, 6, 7));

        let y = String::new();
        let x = join!(async {
            println!("{}", &y);
            1
        })
        .await;
        assert_eq!(x, 1);
    });
}

/// Tests that `join!(…)` behaves "like a function": evaluating its arguments
/// before applying any of its own logic.
///
/// _e.g._, `join!(async_fn(&borrowed), …)` does not consume `borrowed`;
/// and `join!(opt_fut?, …)` does let that `?` refer to the callsite scope.
mod test_join_function_like_value_arg_semantics {
    use super::*;

    async fn async_fn(_: impl Sized) {}

    // no need to _run_ this test, just to compile it.
    fn _join_does_not_unnecessarily_move_mentioned_bindings() {
        let not_copy = vec![()];
        let _ = join!(async_fn(&not_copy)); // should not move `not_copy`
        let _ = &not_copy; // OK
    }

    #[test]
    fn join_lets_control_flow_effects_such_as_try_flow_through() {
        let maybe_fut = None;
        if false {
            *&mut { maybe_fut } = Some(async {});
            loop {}
        }
        assert!(Option::is_none(&try { join!(maybe_fut?, async { unreachable!() }) }));
    }

    #[test]
    fn join_is_able_to_handle_temporaries() {
        let _ = join!(async_fn(&String::from("temporary")));
        let () = block_on(join!(async_fn(&String::from("temporary"))));
    }
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

// just tests by whether or not this compiles
fn _pending_impl_all_auto_traits<T>() {
    use std::panic::{RefUnwindSafe, UnwindSafe};
    fn all_auto_traits<T: Send + Sync + Unpin + UnwindSafe + RefUnwindSafe>() {}

    all_auto_traits::<std::future::Pending<T>>();
}
