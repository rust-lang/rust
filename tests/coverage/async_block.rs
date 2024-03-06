#![feature(coverage_attribute)]
#![feature(noop_waker)]
//@ edition: 2021

fn main() {
    for i in 0..16 {
        let future = async {
            if i >= 12 {
                println!("big");
            } else {
                println!("small");
            }
        };
        executor::block_on(future);
    }
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
