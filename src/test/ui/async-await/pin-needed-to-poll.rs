use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

struct Sleep;

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
    Sleep
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
        self.sleep.poll(cx)
        //~^ ERROR no method named `poll` found for struct `Sleep` in the current scope
    }
}

fn main() {}
