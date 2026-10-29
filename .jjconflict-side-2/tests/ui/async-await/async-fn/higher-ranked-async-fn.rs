//@ aux-build:block-on.rs
//@ edition:2018
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ build-pass (since it ICEs during mono)

extern crate block_on;

use std::future::Future;

async fn f(arg: &i32) {}

async fn func<F>(f: F)
where
    F: for<'a> AsyncFn(&'a i32),
{
    let x: i32 = 0;
    f(&x).await;
}

fn main() {
    block_on::block_on(async {
        // Function
        func(f).await;

        // Regular closure (doesn't capture)
        func(|x: &i32| async {});
    });
}
