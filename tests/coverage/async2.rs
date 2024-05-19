#![feature(coverage_attribute)]
#![feature(noop_waker)]
//@ edition: 2018

fn non_async_func() {
    println!("non_async_func was covered");
    let b = true;
    if b {
        println!("non_async_func println in block");
    }
}

async fn async_func() {
    println!("async_func was covered");
    let b = true;
    if b {
        println!("async_func println in block");
    }
}

async fn async_func_just_println() {
    println!("async_func_just_println was covered");
}

fn main() {
    println!("codecovsample::main");

    non_async_func();

    executor::block_on(async_func());
    executor::block_on(async_func_just_println());
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
