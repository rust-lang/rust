//@ aux-build:arc_wake.rs
//@ edition:2018
//@ run-pass

//@ revisions: default nomiropt
//@[nomiropt]compile-flags: -Z mir-opt-level=0

#![allow(unused_variables)]

// Test the drop order for parameters relative to local variables and
// temporaries created in the tail return expression of the function
// body. In particular, check that this drop order is the same between
// an `async fn` and an ordinary `fn`. See #64512.

extern crate arc_wake;

use arc_wake::ArcWake;
use std::cell::RefCell;
use std::future::Future;
use std::sync::Arc;
use std::rc::Rc;
use std::task::Context;

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

/// Check drop order of temporary "temp" as compared to `x`, `y`, and `z`.
///
/// Expected order:
/// - `z`
/// - temp
/// - `y`
/// - `x`
async fn foo_async(x: D, _y: D) {
    let l = x.1.clone();
    let z = D("z", l.clone());
    l.borrow_mut().push(DropOrder::Function);
    helper_async(&D("temp", l)).await
}

async fn helper_async(v: &D) { }

fn foo_sync(x: D, _y: D) {
    let l = x.1.clone();
    let z = D("z", l.clone());
    l.borrow_mut().push(DropOrder::Function);
    helper_sync(&D("temp", l))
}

fn helper_sync(v: &D) { }

fn assert_drop_order_after_poll<Fut: Future<Output = ()>>(
    f: impl FnOnce(DropOrderListPtr) -> Fut,
    g: impl FnOnce(DropOrderListPtr),
) {
    let empty = Arc::new(EmptyWaker);
    let waker = ArcWake::into_waker(empty);
    let mut cx = Context::from_waker(&waker);

    let actual_order = Rc::new(RefCell::new(Vec::new()));
    let mut fut = Box::pin(f(actual_order.clone()));
    let r = fut.as_mut().poll(&mut cx);

    assert!(match r {
        std::task::Poll::Ready(()) => true,
        _ => false,
    });

    let expected_order = Rc::new(RefCell::new(Vec::new()));
    g(expected_order.clone());

    assert_eq!(*actual_order.borrow(), *expected_order.borrow());
}

fn main() {
    // Free functions (see doc comment on function for what it tests).
    assert_drop_order_after_poll(|l| foo_async(D("x", l.clone()), D("_y", l.clone())),
                                 |l| foo_sync(D("x", l.clone()), D("_y", l.clone())));
}
