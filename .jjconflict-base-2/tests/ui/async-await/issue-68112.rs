//@ edition:2018

use std::{
    cell::RefCell,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

fn require_send(_: impl Send) {}

struct Ready<T>(Option<T>);
impl<T: Unpin> Future for Ready<T> {
    type Output = T;
    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<T> {
        Poll::Ready(self.0.take().unwrap())
    }
}
fn ready<T>(t: T) -> Ready<T> {
    Ready(Some(t))
}

fn make_non_send_future1() -> impl Future<Output = Arc<RefCell<i32>>> {
    ready(Arc::new(RefCell::new(0)))
}

fn test1() {
    let send_fut = async {
        let non_send_fut = make_non_send_future1();
        let _ = non_send_fut.await;
        ready(0).await;
    };
    require_send(send_fut);
    //~^ ERROR future cannot be sent between threads
}

fn test1_no_let() {
    let send_fut = async {
        let _ = make_non_send_future1().await;
        ready(0).await;
    };
    require_send(send_fut);
    //~^ ERROR future cannot be sent between threads
}

async fn ready2<T>(t: T) -> T {
    t
}
fn make_non_send_future2() -> impl Future<Output = Arc<RefCell<i32>>> {
    ready2(Arc::new(RefCell::new(0)))
}

// Ideally this test would have diagnostics similar to the test above, but right
// now it doesn't.
fn test2() {
    let send_fut = async {
        let non_send_fut = make_non_send_future2();
        let _ = non_send_fut.await;
        ready(0).await;
    };
    require_send(send_fut);
    //~^ ERROR `RefCell<i32>` cannot be shared between threads safely
}

fn main() {}
