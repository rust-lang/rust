#![feature(async_fn_traits, async_trait_bounds)]

use std::future::Future;
use std::pin::pin;
use std::task::*;

pub fn block_on<T>(fut: impl Future<Output = T>) -> T {
    let mut fut = pin!(fut);
    let ctx = &mut Context::from_waker(Waker::noop());

    loop {
        match fut.as_mut().poll(ctx) {
            Poll::Pending => {}
            Poll::Ready(t) => break t,
        }
    }
}

async fn call_once(f: impl async FnOnce(DropMe)) {
    f(DropMe("world")).await;
}

#[derive(Debug)]
struct DropMe(&'static str);

impl Drop for DropMe {
    fn drop(&mut self) {
        println!("{}", self.0);
    }
}

pub fn main() {
    block_on(async {
        let b = DropMe("hello");
        let async_closure = async move |a: DropMe| {
            println!("{a:?} {b:?}");
        };
        call_once(async_closure).await;
    });
}
