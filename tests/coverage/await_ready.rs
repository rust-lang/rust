#![feature(coverage_attribute)]
#![feature(custom_inner_attributes)] // for #![rustfmt::skip]
#![feature(noop_waker)]
#![rustfmt::skip]
//@ edition: 2021

#[coverage(off)]
async fn ready() -> u8 { 1 }

async fn await_ready() -> u8 {
    // await should be covered even if the function never yields
    ready()
        .await
}

fn main() {
    let mut future = Box::pin(await_ready());
    executor::block_on(future.as_mut());
}

mod executor {
    use core::future::Future;
    use core::pin::pin;
    use core::task::{Context, Poll, Waker};

    #[coverage(off)]
    pub fn block_on<F: Future>(mut future: F) -> F::Output {
        let mut future = pin!(future);
        let mut context = Context::from_waker(Waker::noop());

        loop {
            if let Poll::Ready(val) = future.as_mut().poll(&mut context) {
                break val;
            }
        }
    }
}
