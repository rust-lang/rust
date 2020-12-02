#![allow(unused_assignments, dead_code)]

// require-rust-edition-2018

async fn c(x: u8) -> u8 {
    if x == 8 {
        1
    } else {
        0
    }
}

async fn d() -> u8 { 1 }

async fn e() -> u8 { 1 } // unused function; executor does not block on `g()`

async fn f() -> u8 { 1 }

async fn foo() -> [bool; 10] { [false; 10] } // unused function; executor does not block on `h()`

pub async fn g(x: u8) {
    match x {
        y if e().await == y => (),
        y if f().await == y => (),
        _ => (),
    }
}

async fn h(x: usize) { // The function signature is counted when called, but the body is not
                       // executed (not awaited) so the open brace has a `0` count (at least when
                       // displayed with `llvm-cov show` in color-mode).
    match x {
        y if foo().await[y] => (),
        _ => (),
    }
}

async fn i(x: u8) { // line coverage is 1, but there are 2 regions:
                    // (a) the function signature, counted when the function is called; and
                    // (b) the open brace for the function body, counted once when the body is
                    // executed asynchronously.
    match x {
        y if c(x).await == y + 1 => { d().await; }
        y if f().await == y + 1 => (),
        _ => (),
    }
}

fn j(x: u8) {
    // non-async versions of `c()`, `d()`, and `f()` to make it similar to async `i()`.
    fn c(x: u8) -> u8 {
        if x == 8 {
            1 // This line appears covered, but the 1-character expression span covering the `1`
              // is not executed. (`llvm-cov show` displays a `^0` below the `1` ). This is because
              // `fn j()` executes the open brace for the funciton body, followed by the function's
              // first executable statement, `match x`. Inner function declarations are not
              // "visible" to the MIR for `j()`, so the code region counts all lines between the
              // open brace and the first statement as executed, which is, in a sense, true.
              // `llvm-cov show` overcomes this kind of situation by showing the actual counts
              // of the enclosed coverages, (that is, the `1` expression was not executed, and
              // accurately displays a `0`).
        } else {
            0
        }
    }
    fn d() -> u8 { 1 }
    fn f() -> u8 { 1 }
    match x {
        y if c(x) == y + 1 => { d(); }
        y if f() == y + 1 => (),
        _ => (),
    }
}

fn k(x: u8) { // unused function
    match x {
        1 => (),
        2 => (),
        _ => (),
    }
}

fn l(x: u8) {
    match x {
        1 => (),
        2 => (),
        _ => (),
    }
}

async fn m(x: u8) -> u8 { x - 1 }

fn main() {
    let _ = g(10);
    let _ = h(9);
    let mut future = Box::pin(i(8));
    j(7);
    l(6);
    let _ = m(5);
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
