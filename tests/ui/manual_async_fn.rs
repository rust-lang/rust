// run-rustfix
// edition:2018
#![warn(clippy::manual_async_fn)]
#![allow(unused)]

use std::future::Future;

fn fut() -> impl Future<Output = i32> {
    async { 42 }
}

fn empty_fut() -> impl Future<Output = ()> {
    async {}
}

fn core_fut() -> impl core::future::Future<Output = i32> {
    async move { 42 }
}

// should be ignored
fn has_other_stmts() -> impl core::future::Future<Output = i32> {
    let _ = 42;
    async move { 42 }
}

// should be ignored
fn not_fut() -> i32 {
    42
}

// should be ignored
async fn already_async() -> impl Future<Output = i32> {
    async { 42 }
}

struct S {}
impl S {
    fn inh_fut() -> impl Future<Output = i32> {
        async {
            // NOTE: this code is here just to check that the indentation is correct in the suggested fix
            let a = 42;
            let b = 21;
            if a < b {
                let c = 21;
                let d = 42;
                if c < d {
                    let _ = 42;
                }
            }
            42
        }
    }

    fn meth_fut(&self) -> impl Future<Output = i32> {
        async { 42 }
    }

    fn empty_fut(&self) -> impl Future<Output = ()> {
        async {}
    }

    // should be ignored
    fn not_fut(&self) -> i32 {
        42
    }

    // should be ignored
    fn has_other_stmts() -> impl core::future::Future<Output = i32> {
        let _ = 42;
        async move { 42 }
    }

    // should be ignored
    async fn already_async(&self) -> impl Future<Output = i32> {
        async { 42 }
    }
}

fn main() {}
