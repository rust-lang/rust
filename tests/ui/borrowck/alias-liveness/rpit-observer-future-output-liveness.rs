//@ edition: 2024
//@ check-pass

// Minimized from some_executor and some_local_executor regressions in #161154.

#![forbid(unsafe_code)]
#![allow(dead_code)]

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

struct FinishedObservation<T>(T);

trait Observer: 'static + Future<Output = FinishedObservation<Self::Value>> {
    type Value;
    fn observe(&self);
}

struct OwnedObserver;
impl Future for OwnedObserver {
    type Output = FinishedObservation<u8>;
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(FinishedObservation(0))
    }
}
impl Observer for OwnedObserver {
    type Value = u8;
    fn observe(&self) {}
}

trait Executor {
    fn spawn<T>(&mut self, _task: T) -> impl Observer;
    fn tick(&mut self);
}

struct Concrete;

impl Executor for Concrete {
    fn spawn<T>(&mut self, _: T) -> impl Observer {
        OwnedObserver
    }
    fn tick(&mut self) {}
}

fn use_after_second_borrow(executor: &mut impl Executor) {
    let observer = executor.spawn(());
    executor.tick();
    observer.observe();
}

fn move_into_second_task(executor: &mut impl Executor) {
    let observer = executor.spawn(());
    executor.spawn(observer).observe();
}

fn main() {}
