//@ aux-build:block-on.rs
//@ edition:2021
//@ run-pass
//@ check-run-results
//@ revisions: call call_once force_once

// call - Call the closure regularly.
// call_once - Call the closure w/ `AsyncFnOnce`, so exercising the by_move shim.
// force_once - Force the closure mode to `FnOnce`, so exercising what was fixed
//   in <https://github.com/rust-lang/rust/pull/123350>.

#![allow(unused_mut)]

extern crate block_on;

#[cfg(any(call, force_once))]
macro_rules! call {
    ($c:expr) => { ($c)() }
}

#[cfg(call_once)]
async fn call_once(f: impl AsyncFnOnce()) {
    f().await
}

#[cfg(call_once)]
macro_rules! call {
    ($c:expr) => { call_once($c) }
}

#[cfg(not(force_once))]
macro_rules! guidance {
    ($c:expr) => { $c }
}

#[cfg(force_once)]
fn infer_fnonce(c: impl AsyncFnOnce()) -> impl AsyncFnOnce() { c }

#[cfg(force_once)]
macro_rules! guidance {
    ($c:expr) => { infer_fnonce($c) }
}

#[derive(Debug)]
struct Drop(&'static str);

impl std::ops::Drop for Drop {
    fn drop(&mut self) {
        println!("{}", self.0);
    }
}

struct S {
    a: i32,
    b: Drop,
    c: Drop,
}

async fn async_main() {
    // Precise capture struct
    {
        let mut s = S { a: 1, b: Drop("fix me up"), c: Drop("untouched") };
        let mut c = guidance!(async || {
            s.a = 2;
            let w = &mut s.b;
            w.0 = "fixed";
        });
        s.c.0 = "uncaptured";
        let fut = call!(c);
        println!("after call");
        fut.await;
        println!("after await");
    }
    println!();

    // Precise capture &mut struct
    {
        let s = &mut S { a: 1, b: Drop("fix me up"), c: Drop("untouched") };
        let mut c = guidance!(async || {
            s.a = 2;
            let w = &mut s.b;
            w.0 = "fixed";
        });
        s.c.0 = "uncaptured";
        let fut = call!(c);
        println!("after call");
        fut.await;
        println!("after await");
    }
    println!();

    // Precise capture struct by move
    {
        let mut s = S { a: 1, b: Drop("fix me up"), c: Drop("untouched") };
        let mut c = guidance!(async move || {
            s.a = 2;
            let w = &mut s.b;
            w.0 = "fixed";
        });
        s.c.0 = "uncaptured";
        let fut = call!(c);
        println!("after call");
        fut.await;
        println!("after await");
    }
    println!();

    // Precise capture &mut struct by move
    {
        let s = &mut S { a: 1, b: Drop("fix me up"), c: Drop("untouched") };
        let mut c = guidance!(async move || {
            s.a = 2;
            let w = &mut s.b;
            w.0 = "fixed";
        });
        // `s` is still captured fully as `&mut S`.
        let fut = call!(c);
        println!("after call");
        fut.await;
        println!("after await");
    }
    println!();

    // Precise capture struct, consume field
    {
        let mut s = S { a: 1, b: Drop("drop first"), c: Drop("untouched") };
        let c = guidance!(async move || {
            s.a = 2;
            drop(s.b);
        });
        s.c.0 = "uncaptured";
        let fut = call!(c);
        println!("after call");
        fut.await;
        println!("after await");
    }
    println!();

    // Precise capture struct by move, consume field
    {
        let mut s = S { a: 1, b: Drop("drop first"), c: Drop("untouched") };
        let c = guidance!(async move || {
            s.a = 2;
            drop(s.b);
        });
        s.c.0 = "uncaptured";
        let fut = call!(c);
        println!("after call");
        fut.await;
        println!("after await");
    }
}

fn main() {
    block_on::block_on(async_main());
}
