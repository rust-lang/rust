// Verifies that coroutines (i.e., coroutines, async functions, gen functions,
// and async gen functions) can be called through their trait objects, and that
// functions with coroutine types as argument types can be called through
// function pointers.
//
//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ edition: 2024
//@ compile-flags: -Ctarget-feature=-crt-static -Ccodegen-units=1 -Clto -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=cfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

#![feature(coroutines, stmt_expr_attributes)]
#![feature(coroutine_trait)]
#![feature(gen_blocks)]
#![feature(async_iterator)]

use std::async_iter::AsyncIterator;
use std::ops::{Coroutine, CoroutineState};
use std::pin::{Pin, pin};
use std::task::{Context, Poll, Waker};

// The async fn coroutine is transformed into <dyn Future<Output = i32> as Future>::poll
async fn async_fn() -> i32 {
    3
}

// The gen fn coroutine is transformed into <dyn Iterator<Item = i32> as Iterator>::next
gen fn gen_fn() -> i32 {
    yield 5;
}

// The async gen fn coroutine is transformed into
// <dyn AsyncIterator<Item = i32> as AsyncIterator>::poll_next.
async gen fn async_gen_fn() -> i32 {
    yield 6;
}

fn generic_coroutine<T: Future<Output = i32>>(_: T) -> i32 {
    7
}

fn main() {
    // Coroutines
    // The coroutine is transformed into
    // <dyn Coroutine<i32, Yield = i32, Return = i32> as Coroutine<i32>>::resume.
    let coro = #[coroutine]
    |_: i32| {
        yield 1;
        2
    };
    let mut abstract_coro: Pin<&mut dyn Coroutine<i32, Yield = i32, Return = i32>> = pin!(coro);
    // The virtual method call is transformed into
    // <dyn Coroutine<i32, Yield = i32, Return = i32> as Coroutine<i32>>::resume.
    assert_eq!(abstract_coro.as_mut().resume(1), CoroutineState::Yielded(1));
    // The virtual method call is transformed into
    // <dyn Coroutine<i32, Yield = i32, Return = i32> as Coroutine<i32>>::resume.
    assert_eq!(abstract_coro.as_mut().resume(2), CoroutineState::Complete(2));

    // Async fn coroutines
    let f: fn() -> Pin<Box<dyn Future<Output = i32>>> =
        std::hint::black_box(|| Box::pin(async_fn()));
    // The virtual method call is transformed into <dyn Future<Output = i32> as Future>::poll
    assert_eq!(f().as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Ready(3));

    // Async block coroutines
    // The async block coroutine is transformed into <dyn Future<Output = i32> as Future>::poll
    let g = async {
        f().await;
        4
    };
    assert_eq!(pin!(g).poll(&mut Context::from_waker(Waker::noop())), Poll::Ready(4));

    // Gen fn coroutines
    let f: fn() -> Box<dyn Iterator<Item = i32>> = std::hint::black_box(|| Box::new(gen_fn()));
    // The virtual method call is transformed into <dyn Iterator<Item = i32> as Iterator>::next
    assert_eq!(f().next(), Some(5));

    // Async gen fn coroutines
    let f: fn() -> Pin<Box<dyn AsyncIterator<Item = i32>>> =
        std::hint::black_box(|| Box::pin(async_gen_fn()));
    // The virtual method call is transformed into
    // <dyn AsyncIterator<Item = i32> as AsyncIterator>::poll_next.
    assert_eq!(
        f().as_mut().poll_next(&mut Context::from_waker(Waker::noop())),
        Poll::Ready(Some(6))
    );

    // Concrete coroutine types
    // The concrete coroutine type, and not a trait object of it, is used in the signature, so
    // ty::CoroutineWitness is encoded (see issue #111184)
    let f: fn(_) -> i32 = std::hint::black_box(generic_coroutine);
    assert_eq!(f(async_fn()), 7);
}
