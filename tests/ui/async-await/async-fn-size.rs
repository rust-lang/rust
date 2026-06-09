//@ run-pass
//@ aux-build:arc_wake.rs
//@ edition:2018

extern crate arc_wake;

use std::pin::Pin;
use std::future::Future;
use std::sync::{
    Arc,
    atomic::{self, AtomicUsize},
};
use std::task::{Context, Poll};
use arc_wake::ArcWake;

struct Counter {
    wakes: AtomicUsize,
}

impl ArcWake for Counter {
    fn wake(self: Arc<Self>) {
        Self::wake_by_ref(&self)
    }
    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.wakes.fetch_add(1, atomic::Ordering::SeqCst);
    }
}

struct WakeOnceThenComplete(bool, u8);

impl Future for WakeOnceThenComplete {
    type Output = u8;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u8> {
        if self.0 {
            Poll::Ready(self.1)
        } else {
            cx.waker().wake_by_ref();
            self.0 = true;
            Poll::Pending
        }
    }
}

fn wait(fut: impl Future<Output = u8>) -> u8 {
    let mut fut = Box::pin(fut);
    let counter = Arc::new(Counter { wakes: AtomicUsize::new(0) });
    let waker = ArcWake::into_waker(counter.clone());
    let mut cx = Context::from_waker(&waker);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(out) => return out,
            Poll::Pending => (),
        }
    }
}

fn base() -> WakeOnceThenComplete { WakeOnceThenComplete(false, 1) }

async fn await1_level1() -> u8 {
    base().await
}

async fn await2_level1() -> u8 {
    base().await + base().await
}

async fn await3_level1() -> u8 {
    base().await + base().await + base().await
}

async fn await3_level2() -> u8 {
    await3_level1().await + await3_level1().await + await3_level1().await
}

async fn await3_level3() -> u8 {
    await3_level2().await + await3_level2().await + await3_level2().await
}

async fn await3_level4() -> u8 {
    await3_level3().await + await3_level3().await + await3_level3().await
}

async fn await3_level5() -> u8 {
    await3_level4().await + await3_level4().await + await3_level4().await
}

fn main() {
    assert_eq!(2, std::mem::size_of_val(&base()));
    assert_eq!(3, std::mem::size_of_val(&await1_level1()));
    assert_eq!(4, std::mem::size_of_val(&await2_level1()));
    assert_eq!(5, std::mem::size_of_val(&await3_level1()));
    assert_eq!(8, std::mem::size_of_val(&await3_level2()));
    assert_eq!(11, std::mem::size_of_val(&await3_level3()));
    assert_eq!(14, std::mem::size_of_val(&await3_level4()));
    assert_eq!(17, std::mem::size_of_val(&await3_level5()));

    assert_eq!(1,   wait(base()));
    assert_eq!(1,   wait(await1_level1()));
    assert_eq!(2,   wait(await2_level1()));
    assert_eq!(3,   wait(await3_level1()));
    assert_eq!(9,   wait(await3_level2()));
    assert_eq!(27,  wait(await3_level3()));
    assert_eq!(81,  wait(await3_level4()));
    assert_eq!(243, wait(await3_level5()));
}
