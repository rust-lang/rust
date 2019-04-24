// aux-build:arc_wake.rs
// edition:2018
// run-pass

#![allow(unused_variables)]
#![feature(async_await, await_macro)]

extern crate arc_wake;

use arc_wake::ArcWake;
use std::cell::RefCell;
use std::future::Future;
use std::marker::PhantomData;
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

/// Check that unused bindings are dropped after the function is polled.
async fn foo(x: D, _y: D) {
    x.1.borrow_mut().push(DropOrder::Function);
}

/// Check that underscore patterns are dropped after the function is polled.
async fn bar(x: D, _: D) {
    x.1.borrow_mut().push(DropOrder::Function);
}

/// Check that underscore patterns within more complex patterns are dropped after the function
/// is polled.
async fn baz((x, _): (D, D)) {
    x.1.borrow_mut().push(DropOrder::Function);
}

/// Check that underscore and unused bindings within and outwith more complex patterns are dropped
/// after the function is polled.
async fn foobar(x: D, (a, _, _c): (D, D, D), _: D, _y: D) {
    x.1.borrow_mut().push(DropOrder::Function);
}

struct Foo;

impl Foo {
    /// Check that unused bindings are dropped after the method is polled.
    async fn foo(x: D, _y: D) {
        x.1.borrow_mut().push(DropOrder::Function);
    }

    /// Check that underscore patterns are dropped after the method is polled.
    async fn bar(x: D, _: D) {
        x.1.borrow_mut().push(DropOrder::Function);
    }

    /// Check that underscore patterns within more complex patterns are dropped after the method
    /// is polled.
    async fn baz((x, _): (D, D)) {
        x.1.borrow_mut().push(DropOrder::Function);
    }

    /// Check that underscore and unused bindings within and outwith more complex patterns are
    /// dropped after the method is polled.
    async fn foobar(x: D, (a, _, _c): (D, D, D), _: D, _y: D) {
        x.1.borrow_mut().push(DropOrder::Function);
    }
}

struct Bar<'a>(PhantomData<&'a ()>);

impl<'a> Bar<'a> {
    /// Check that unused bindings are dropped after the method with self is polled.
    async fn foo(&'a self, x: D, _y: D) {
        x.1.borrow_mut().push(DropOrder::Function);
    }

    /// Check that underscore patterns are dropped after the method with self is polled.
    async fn bar(&'a self, x: D, _: D) {
        x.1.borrow_mut().push(DropOrder::Function);
    }

    /// Check that underscore patterns within more complex patterns are dropped after the method
    /// with self is polled.
    async fn baz(&'a self, (x, _): (D, D)) {
        x.1.borrow_mut().push(DropOrder::Function);
    }

    /// Check that underscore and unused bindings within and outwith more complex patterns are
    /// dropped after the method with self is polled.
    async fn foobar(&'a self, x: D, (a, _, _c): (D, D, D), _: D, _y: D) {
        x.1.borrow_mut().push(DropOrder::Function);
    }
}

fn assert_drop_order_after_poll<Fut: Future<Output = ()>>(
    f: impl FnOnce(DropOrderListPtr) -> Fut,
    expected_order: &[DropOrder],
) {
    let empty = Arc::new(EmptyWaker);
    let waker = ArcWake::into_waker(empty);
    let mut cx = Context::from_waker(&waker);

    let actual_order = Rc::new(RefCell::new(Vec::new()));
    let mut fut = Box::pin(f(actual_order.clone()));
    let _ = fut.as_mut().poll(&mut cx);

    assert_eq!(*actual_order.borrow(), expected_order);
}

fn main() {
    use DropOrder::*;

    // At time of writing (23/04/19), the `bar` and `foobar` tests do not output the same order as
    // the equivalent non-async functions. This is because the drop order of captured variables
    // doesn't match the drop order of arguments in a function.

    // Free functions (see doc comment on function for what it tests).
    assert_drop_order_after_poll(|l| foo(D("x", l.clone()), D("_y", l.clone())),
                                 &[Function, Val("_y"), Val("x")]);
    assert_drop_order_after_poll(|l| bar(D("x", l.clone()), D("_", l.clone())),
                                 &[Function, Val("x"), Val("_")]);
    assert_drop_order_after_poll(|l| baz((D("x", l.clone()), D("_", l.clone()))),
                                 &[Function, Val("x"), Val("_")]);
    assert_drop_order_after_poll(|l| {
        foobar(
            D("x", l.clone()),
            (D("a", l.clone()), D("_", l.clone()), D("_c", l.clone())),
            D("_", l.clone()),
            D("_y", l.clone()),
        )
    }, &[Function, Val("_y"), Val("_c"), Val("a"), Val("x"), Val("_"), Val("_")]);

    // Methods w/out self (see doc comment on function for what it tests).
    assert_drop_order_after_poll(|l| Foo::foo(D("x", l.clone()), D("_y", l.clone())),
                                 &[Function, Val("_y"), Val("x")]);
    assert_drop_order_after_poll(|l| Foo::bar(D("x", l.clone()), D("_", l.clone())),
                                 &[Function, Val("x"), Val("_")]);
    assert_drop_order_after_poll(|l| Foo::baz((D("x", l.clone()), D("_", l.clone()))),
                                 &[Function, Val("x"), Val("_")]);
    assert_drop_order_after_poll(|l| {
        Foo::foobar(
            D("x", l.clone()),
            (D("a", l.clone()), D("_", l.clone()), D("_c", l.clone())),
            D("_", l.clone()),
            D("_y", l.clone()),
        )
    }, &[Function, Val("_y"), Val("_c"), Val("a"), Val("x"), Val("_"), Val("_")]);

    // Methods (see doc comment on function for what it tests).
    let b = Bar(Default::default());
    assert_drop_order_after_poll(|l| b.foo(D("x", l.clone()), D("_y", l.clone())),
                                 &[Function, Val("_y"), Val("x")]);
    assert_drop_order_after_poll(|l| b.bar(D("x", l.clone()), D("_", l.clone())),
                                 &[Function, Val("x"), Val("_")]);
    assert_drop_order_after_poll(|l| b.baz((D("x", l.clone()), D("_", l.clone()))),
                                 &[Function, Val("x"), Val("_")]);
    assert_drop_order_after_poll(|l| {
        b.foobar(
            D("x", l.clone()),
            (D("a", l.clone()), D("_", l.clone()), D("_c", l.clone())),
            D("_", l.clone()),
            D("_y", l.clone()),
        )
    }, &[Function, Val("_y"), Val("_c"), Val("a"), Val("x"), Val("_"), Val("_")]);
}
