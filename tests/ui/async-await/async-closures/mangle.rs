// aux-build:block-on.rs
// edition:2021
// build-pass
// revisions: v0 legacy
//[v0] compile-flags: -Csymbol-mangling-version=v0
//[legacy] compile-flags: -Csymbol-mangling-version=legacy -Zunstable-options

// FIXME(async_closures): When `fn_sig_for_fn_abi` is fixed, remove this.
// ignore-pass (test emits codegen-time warnings)

#![feature(async_closure, noop_waker, async_fn_traits)]

extern crate block_on;

use std::future::Future;
use std::ops::{AsyncFnMut, AsyncFnOnce};
use std::pin::pin;
use std::task::*;

async fn call_mut(f: &mut impl AsyncFnMut()) {
    f().await;
}

async fn call_once(f: impl AsyncFnOnce()) {
    f().await;
}

fn main() {
    block_on::block_on(async {
        let mut async_closure = async move || {
            println!("called");
        };
        call_mut(&mut async_closure).await;
        call_once(async_closure).await;
    });
}
