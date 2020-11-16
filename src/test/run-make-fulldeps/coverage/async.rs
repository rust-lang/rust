#![allow(unused_assignments)]

// require-rust-edition-2018

async fn f() -> u8 { 1 }

async fn foo() -> [bool; 10] { [false; 10] }

pub async fn g(x: u8) {
    match x {
        y if f().await == y => (),
        _ => (),
    }
}

// #78366: check the reference to the binding is recorded even if the binding is not autorefed

async fn h(x: usize) {
    match x {
        y if foo().await[y] => (),
        _ => (),
    }
}

async fn i(x: u8) {
    match x {
        y if f().await == y + 1 => (),
        _ => (),
    }
}

fn main() {
    let _ = g(10);
    let _ = h(9);
    let mut future = Box::pin(i(8));
    executor::block_on(future.as_mut());
}

mod executor {
    use core::{
        future::Future,
        pin::Pin,
        task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
    };

    pub fn block_on<F: Future>(mut future: F) -> F::Output {
        let mut future = unsafe { Pin::new_unchecked(&mut future) };

        static VTABLE: RawWakerVTable = RawWakerVTable::new(
            |_| unimplemented!("clone"),
            |_| unimplemented!("wake"),
            |_| unimplemented!("wake_by_ref"),
            |_| (),
        );
        let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) };
        let mut context = Context::from_waker(&waker);

        loop {
            if let Poll::Ready(val) = future.as_mut().poll(&mut context) {
                break val;
            }
        }
    }
}
