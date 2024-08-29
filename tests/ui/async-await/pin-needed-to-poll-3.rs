use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};


struct FutureWrapper<F> {
    fut: F,
}

impl<F> Future for FutureWrapper<F>
where
    F: Future,
{
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let res = self.fut.poll(cx);
        //~^ ERROR no method named `poll` found for type parameter `F` in the current scope
        res
    }
}

fn main() {}
