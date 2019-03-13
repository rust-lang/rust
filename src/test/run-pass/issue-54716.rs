// aux-build:arc_wake.rs
// edition:2018
// run-pass

#![allow(unused_variables)]
#![feature(async_await, await_macro, futures_api)]

extern crate arc_wake;

use arc_wake::ArcWake;
use std::cell::RefCell;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
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

struct D(&'static str, Arc<RefCell<Vec<DropOrder>>>);

impl Drop for D {
    fn drop(&mut self) {
        self.1.borrow_mut().push(DropOrder::Val(self.0));
    }
}

async fn foo(x: D, _y: D) {
    x.1.borrow_mut().push(DropOrder::Function);
}

async fn bar(x: D, _: D) {
    x.1.borrow_mut().push(DropOrder::Function);
}

async fn baz((x, _): (D, D)) {
    x.1.borrow_mut().push(DropOrder::Function);
}

async fn foobar(x: D, (a, _, _c): (D, D, D), _: D, _y: D) {
    x.1.borrow_mut().push(DropOrder::Function);
}

struct Foo;

impl Foo {
    async fn foo(x: D, _y: D) {
        x.1.borrow_mut().push(DropOrder::Function);
    }

    async fn bar(x: D, _: D) {
        x.1.borrow_mut().push(DropOrder::Function);
    }

    async fn baz((x, _): (D, D)) {
        x.1.borrow_mut().push(DropOrder::Function);
    }

    async fn foobar(x: D, (a, _, _c): (D, D, D), _: D, _y: D) {
        x.1.borrow_mut().push(DropOrder::Function);
    }
}

struct Bar<'a>(PhantomData<&'a ()>);

impl<'a> Bar<'a> {
    async fn foo(&'a self, x: D, _y: D) {
        x.1.borrow_mut().push(DropOrder::Function);
    }

    async fn bar(&'a self, x: D, _: D) {
        x.1.borrow_mut().push(DropOrder::Function);
    }

    async fn baz(&'a self, (x, _): (D, D)) {
        x.1.borrow_mut().push(DropOrder::Function);
    }

    async fn foobar(&'a self, x: D, (a, _, _c): (D, D, D), _: D, _y: D) {
        x.1.borrow_mut().push(DropOrder::Function);
    }
}

fn main() {
    let empty = Arc::new(EmptyWaker);
    let waker = ArcWake::into_waker(empty);
    let mut cx = Context::from_waker(&waker);

    use DropOrder::*;

    // Currently, the `bar` and `foobar` tests do not output the same order as the equivalent
    // non-async functions. This is because the drop order of captured variables doesn't match the
    // drop order of arguments in a function.

    // Free functions

    let af = Arc::new(RefCell::new(Vec::new()));
    let mut fut = Box::pin(foo(D("x", af.clone()), D("_y", af.clone())));
    let _ = fut.as_mut().poll(&mut cx);
    assert_eq!(*af.borrow(), &[Function, Val("_y"), Val("x")]);

    let af = Arc::new(RefCell::new(Vec::new()));
    let mut fut = Box::pin(bar(D("x", af.clone()), D("_", af.clone())));
    let _ = fut.as_mut().poll(&mut cx);
    assert_eq!(*af.borrow(), &[Function, Val("x"), Val("_")]);

    let af = Arc::new(RefCell::new(Vec::new()));
    let mut fut = Box::pin(baz((D("x", af.clone()), D("_", af.clone()))));
    let _ = fut.as_mut().poll(&mut cx);
    assert_eq!(*af.borrow(), &[Function, Val("x"), Val("_")]);

    let af = Arc::new(RefCell::new(Vec::new()));
    let mut fut = Box::pin(foobar(
        D("x", af.clone()),
        (D("a", af.clone()), D("_", af.clone()), D("_c", af.clone())),
        D("_", af.clone()),
        D("_y", af.clone()),
    ));
    let _ = fut.as_mut().poll(&mut cx);
    assert_eq!(*af.borrow(), &[
       Function, Val("_y"), Val("_c"), Val("a"), Val("x"), Val("_"), Val("_"),
    ]);

    // Methods w/out self

    let af = Arc::new(RefCell::new(Vec::new()));
    let mut fut = Box::pin(Foo::foo(D("x", af.clone()), D("_y", af.clone())));
    let _ = fut.as_mut().poll(&mut cx);
    assert_eq!(*af.borrow(), &[Function, Val("_y"), Val("x")]);

    let af = Arc::new(RefCell::new(Vec::new()));
    let mut fut = Box::pin(Foo::bar(D("x", af.clone()), D("_", af.clone())));
    let _ = fut.as_mut().poll(&mut cx);
    assert_eq!(*af.borrow(), &[Function, Val("x"), Val("_")]);

    let af = Arc::new(RefCell::new(Vec::new()));
    let mut fut = Box::pin(Foo::baz((D("x", af.clone()), D("_", af.clone()))));
    let _ = fut.as_mut().poll(&mut cx);
    assert_eq!(*af.borrow(), &[Function, Val("x"), Val("_")]);

    let af = Arc::new(RefCell::new(Vec::new()));
    let mut fut = Box::pin(Foo::foobar(
        D("x", af.clone()),
        (D("a", af.clone()), D("_", af.clone()), D("_c", af.clone())),
        D("_", af.clone()),
        D("_y", af.clone()),
    ));
    let _ = fut.as_mut().poll(&mut cx);
    assert_eq!(*af.borrow(), &[
       Function, Val("_y"), Val("_c"), Val("a"), Val("x"), Val("_"), Val("_"),
    ]);

    // Methods

    let b = Bar(Default::default());

    let af = Arc::new(RefCell::new(Vec::new()));
    let mut fut = Box::pin(b.foo(D("x", af.clone()), D("_y", af.clone())));
    let _ = fut.as_mut().poll(&mut cx);
    assert_eq!(*af.borrow(), &[Function, Val("_y"), Val("x")]);

    let af = Arc::new(RefCell::new(Vec::new()));
    let mut fut = Box::pin(b.bar(D("x", af.clone()), D("_", af.clone())));
    let _ = fut.as_mut().poll(&mut cx);
    assert_eq!(*af.borrow(), &[Function, Val("x"), Val("_")]);

    let af = Arc::new(RefCell::new(Vec::new()));
    let mut fut = Box::pin(b.baz((D("x", af.clone()), D("_", af.clone()))));
    let _ = fut.as_mut().poll(&mut cx);
    assert_eq!(*af.borrow(), &[Function, Val("x"), Val("_")]);

    let af = Arc::new(RefCell::new(Vec::new()));
    let mut fut = Box::pin(b.foobar(
        D("x", af.clone()),
        (D("a", af.clone()), D("_", af.clone()), D("_c", af.clone())),
        D("_", af.clone()),
        D("_y", af.clone()),
    ));
    let _ = fut.as_mut().poll(&mut cx);
    assert_eq!(*af.borrow(), &[
       Function, Val("_y"), Val("_c"), Val("a"), Val("x"), Val("_"), Val("_"),
    ]);
}
