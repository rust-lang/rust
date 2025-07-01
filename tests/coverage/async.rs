#![feature(coverage_attribute)]
#![feature(custom_inner_attributes)] // for #![rustfmt::skip]
#![allow(unused_assignments, dead_code)]
#![rustfmt::skip]
//@ edition: 2018
//@ compile-flags: -Copt-level=1

//@ aux-build: executor.rs
extern crate executor;

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

async fn g(x: u8) {
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
            1
        } else {
            0
        }
    }
    fn d() -> u8 { 1 } // inner function is defined in-line, but the function is not executed
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
