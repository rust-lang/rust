#![feature(pin_macro)]

use std::future::*;
use std::marker::PhantomPinned;
use std::pin::*;
use std::ptr;
use std::task::*;

struct Delay {
    delay: usize,
}

impl Delay {
    fn new(delay: usize) -> Self {
        Delay { delay }
    }
}

impl Future for Delay {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        if self.delay > 0 {
            self.delay -= 1;
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    }
}

async fn do_stuff() {
    (&mut Delay::new(1)).await;
}

// Same thing implemented by hand
struct DoStuff {
    state: usize,
    delay: Delay,
    delay_ref: *mut Delay,
    _marker: PhantomPinned,
}

impl DoStuff {
    fn new() -> Self {
        DoStuff {
            state: 0,
            delay: Delay::new(1),
            delay_ref: ptr::null_mut(),
            _marker: PhantomPinned,
        }
    }
}

impl Future for DoStuff {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        unsafe {
            let this = self.get_unchecked_mut();
            match this.state {
                0 => {
                    // Set up self-ref.
                    this.delay_ref = &mut this.delay;
                    // Move to next state.
                    this.state = 1;
                    Poll::Pending
                }
                1 => {
                    let delay = &mut *this.delay_ref;
                    Pin::new_unchecked(delay).poll(cx)
                }
                _ => unreachable!(),
            }
        }
    }
}

fn run_fut<T>(fut: impl Future<Output = T>) -> T {
    use std::sync::Arc;

    struct MyWaker;
    impl Wake for MyWaker {
        fn wake(self: Arc<Self>) {
            unimplemented!()
        }
    }

    let waker = Waker::from(Arc::new(MyWaker));
    let mut context = Context::from_waker(&waker);

    let mut pinned = pin!(fut);
    loop {
        match pinned.as_mut().poll(&mut context) {
            Poll::Pending => continue,
            Poll::Ready(v) => return v,
        }
    }
}

fn main() {
    run_fut(do_stuff());
    run_fut(DoStuff::new());
}
