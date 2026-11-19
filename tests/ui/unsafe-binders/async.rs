//@ compile-flags: -Znext-solver
//@ edition: 2024
//@ check-pass

// Unsafe binders can be held across `.await`, and the future stays `Send`.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::unsafe_binder::{unwrap_binder, wrap_binder};

async fn yield_now() {}

async fn across_await(x: &u8) -> u8 {
    let b: unsafe<'a> &'a u8 = unsafe { wrap_binder!(x) };
    yield_now().await;
    unsafe { *unwrap_binder!(b) }
}

fn is_send<T: Send>(_: T) {}

fn send(x: &u8) {
    is_send(across_await(x));
}

fn main() {}
