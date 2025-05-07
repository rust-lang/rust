// Verifies that we can call dynamic coroutines

//@ revisions: cfi kcfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ edition: 2024
//@ [cfi] needs-sanitizer-cfi
//@ [kcfi] needs-sanitizer-kcfi
//@ compile-flags: -C target-feature=-crt-static
//@ [cfi] compile-flags: -C codegen-units=1 -C lto -C prefer-dynamic=off -C opt-level=0
//@ [cfi] compile-flags: -Z sanitizer=cfi
//@ [kcfi] compile-flags: -Z sanitizer=kcfi
//@ [kcfi] compile-flags: -C panic=abort -Z panic-abort-tests -C prefer-dynamic=off
//@ compile-flags: --test
//@ run-pass

#![feature(coroutines, stmt_expr_attributes)]
#![feature(coroutine_trait)]
#![feature(gen_blocks)]
#![feature(async_iterator)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::{pin, Pin};
use std::task::{Context, Poll, Waker};
use std::async_iter::AsyncIterator;

#[test]
fn general_coroutine() {
    let coro = #[coroutine] |x: i32| {
        yield x;
        "done"
    };
    let mut abstract_coro: Pin<&mut dyn Coroutine<i32,Yield=i32,Return=&'static str>> = pin!(coro);
    assert_eq!(abstract_coro.as_mut().resume(2), CoroutineState::Yielded(2));
    assert_eq!(abstract_coro.as_mut().resume(0), CoroutineState::Complete("done"));
}

async fn async_fn() {}

#[test]
fn async_coroutine() {
    let f: fn() -> Pin<Box<dyn Future<Output = ()>>> = || Box::pin(async_fn());
    let _ = async { f().await; };
    assert_eq!(f().as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Ready(()));
}

async gen fn async_gen_fn() -> u8 {
    yield 5;
}

#[test]
fn async_gen_coroutine() {
    let f: fn() -> Pin<Box<dyn AsyncIterator<Item = u8>>> = || Box::pin(async_gen_fn());
    assert_eq!(f().as_mut().poll_next(&mut Context::from_waker(Waker::noop())),
               Poll::Ready(Some(5)));
}

gen fn gen_fn() -> u8 {
    yield 6;
}

#[test]
fn gen_coroutine() {
    let f: fn() -> Box<dyn Iterator<Item = u8>> = || Box::new(gen_fn());
    assert_eq!(f().next(), Some(6));
}
