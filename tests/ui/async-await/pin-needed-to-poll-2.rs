use std::{
    future::Future,
    pin::Pin,
    marker::Unpin,
    task::{Context, Poll},
};

struct Sleep(std::marker::PhantomPinned);

impl Future for Sleep {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(())
    }
}

impl Drop for Sleep {
    fn drop(&mut self) {}
}

fn sleep() -> Sleep {
    Sleep(std::marker::PhantomPinned)
}


struct MyFuture {
    sleep: Sleep,
}

impl MyFuture {
    fn new() -> Self {
        Self {
            sleep: sleep(),
        }
    }
}

impl Future for MyFuture {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.sleep).poll(cx)
        //~^ ERROR `PhantomPinned` cannot be unpinned
    }
}

fn main() {}
