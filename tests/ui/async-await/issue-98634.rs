// edition: 2021

use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll, Waker},
};

pub struct StructAsync<F: Fn() -> Pin<Box<dyn Future<Output = ()>>>> {
    pub callback: F,
}

impl<F> Future for StructAsync<F>
where
    F: Fn() -> Pin<Box<dyn Future<Output = ()>>>,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

async fn callback() {}

struct Runtime;

fn waker() -> &'static Waker {
    todo!()
}

impl Runtime {
    #[track_caller]
    pub fn block_on<F: Future>(&self, mut future: F) -> F::Output {
        loop {
            unsafe {
                Pin::new_unchecked(&mut future).poll(&mut Context::from_waker(waker()));
            }
        }
    }
}

fn main() {
    Runtime.block_on(async {
        StructAsync { callback }.await;
        //~^ ERROR expected `callback` to be a fn item that returns `Pin<Box<dyn Future<Output = ()>>>`, but it returns `impl Future<Output = ()>`
        //~| ERROR expected `callback` to be a fn item that returns `Pin<Box<dyn Future<Output = ()>>>`, but it returns `impl Future<Output = ()>`
        //~| ERROR expected `callback` to be a fn item that returns `Pin<Box<dyn Future<Output = ()>>>`, but it returns `impl Future<Output = ()>`
    });
}
