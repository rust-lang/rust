//@ aux-build:arc_wake.rs
//@ edition:2018
//@ run-pass

#![deny(dead_code)]
#![allow(unused_variables)]
#![allow(unused_must_use)]
#![allow(path_statements)]

// Test that the drop order for locals in a fn and async fn matches up.
extern crate arc_wake;

use arc_wake::ArcWake;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Context, Poll};

struct EmptyWaker;

impl ArcWake for EmptyWaker {
    fn wake(self: Arc<Self>) {}
}

#[derive(Debug, Eq, PartialEq)]
enum DropOrder {
    Function,
    Val(&'static str),
}

type DropOrderListPtr = Rc<RefCell<Vec<DropOrder>>>;

struct D(&'static str, DropOrderListPtr);

impl Drop for D {
    fn drop(&mut self) {
        self.1.borrow_mut().push(DropOrder::Val(self.0));
    }
}

struct NeverReady;

impl Future for NeverReady {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        Poll::Pending
    }
}

async fn simple_variable_declaration_async(l: DropOrderListPtr) {
    l.borrow_mut().push(DropOrder::Function);
    let x = D("x", l.clone());
    let y = D("y", l.clone());
    NeverReady.await;
}

fn simple_variable_declaration_sync(l: DropOrderListPtr) {
    l.borrow_mut().push(DropOrder::Function);
    let x = D("x", l.clone());
    let y = D("y", l.clone());
}

async fn varable_completely_contained_within_block_async(l: DropOrderListPtr) {
    l.borrow_mut().push(DropOrder::Function);
    async {
        let x = D("x", l.clone());
    }
        .await;
    let y = D("y", l.clone());
    NeverReady.await;
}

fn varable_completely_contained_within_block_sync(l: DropOrderListPtr) {
    l.borrow_mut().push(DropOrder::Function);
    {
        let x = D("x", l.clone());
    }
    let y = D("y", l.clone());
}

async fn variables_moved_into_separate_blocks_async(l: DropOrderListPtr) {
    l.borrow_mut().push(DropOrder::Function);
    let x = D("x", l.clone());
    let y = D("y", l.clone());
    async move { x }.await;
    async move { y }.await;
    NeverReady.await;
}

fn variables_moved_into_separate_blocks_sync(l: DropOrderListPtr) {
    l.borrow_mut().push(DropOrder::Function);
    let x = D("x", l.clone());
    let y = D("y", l.clone());
    {
        x
    };
    {
        y
    };
}

async fn variables_moved_into_same_block_async(l: DropOrderListPtr) {
    l.borrow_mut().push(DropOrder::Function);
    let x = D("x", l.clone());
    let y = D("y", l.clone());
    async move {
        x;
        y;
    };
    NeverReady.await;
}

fn variables_moved_into_same_block_sync(l: DropOrderListPtr) {
    l.borrow_mut().push(DropOrder::Function);
    let x = D("x", l.clone());
    let y = D("y", l.clone());
    {
        x;
        y;
    };
    return;
}

async fn move_after_current_await_doesnt_affect_order(l: DropOrderListPtr) {
    l.borrow_mut().push(DropOrder::Function);
    let x = D("x", l.clone());
    let y = D("y", l.clone());
    NeverReady.await;
    async move {
        x;
        y;
    };
}

fn assert_drop_order_after_cancel<Fut: Future<Output = ()>>(
    f: impl FnOnce(DropOrderListPtr) -> Fut,
    g: impl FnOnce(DropOrderListPtr),
) {
    let empty = Arc::new(EmptyWaker);
    let waker = ArcWake::into_waker(empty);
    let mut cx = Context::from_waker(&waker);

    let actual_order = Rc::new(RefCell::new(Vec::new()));
    let mut fut = Box::pin(f(actual_order.clone()));
    let _ = fut.as_mut().poll(&mut cx);
    drop(fut);

    let expected_order = Rc::new(RefCell::new(Vec::new()));
    g(expected_order.clone());
    assert_eq!(*actual_order.borrow(), *expected_order.borrow());
}

fn main() {
    assert_drop_order_after_cancel(
        simple_variable_declaration_async,
        simple_variable_declaration_sync,
    );
    assert_drop_order_after_cancel(
        varable_completely_contained_within_block_async,
        varable_completely_contained_within_block_sync,
    );
    assert_drop_order_after_cancel(
        variables_moved_into_separate_blocks_async,
        variables_moved_into_separate_blocks_sync,
    );
    assert_drop_order_after_cancel(
        variables_moved_into_same_block_async,
        variables_moved_into_same_block_sync,
    );
    assert_drop_order_after_cancel(
        move_after_current_await_doesnt_affect_order,
        simple_variable_declaration_sync,
    );
}
