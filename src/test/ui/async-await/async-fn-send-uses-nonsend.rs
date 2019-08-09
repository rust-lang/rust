// build-pass (FIXME(62277): could be check-pass?)
// edition:2018
// compile-flags: --crate-type lib

#![feature(async_await)]

use std::{
    cell::RefCell,
    fmt::Debug,
    rc::Rc,
};

fn non_sync() -> impl Debug { RefCell::new(()) }

fn non_send() -> impl Debug { Rc::new(()) }

fn take_ref<T>(_: &T) {}

async fn fut() {}

async fn fut_arg<T>(_: T) {}

async fn still_send() {
    fut().await;
    println!("{:?} {:?}", non_send(), non_sync());
    fut().await;
    drop(non_send());
    drop(non_sync());
    fut().await;
    fut_arg(non_sync()).await;

    // Note: all temporaries in `if let` and `match` scrutinee
    // are dropped at the *end* of the blocks, so using `non_send()`
    // in either of those positions with an await in the middle will
    // cause a `!Send` future. It might be nice in the future to allow
    // this for `Copy` types, since they can be "dropped" early without
    // affecting the end user.
    if let Some(_) = Some(non_sync()) {
        fut().await;
    }
    match Some(non_sync()) {
        Some(_) => fut().await,
        None => fut().await,
    }

    let _ = non_send();
    fut().await;

    {
        let _x = non_send();
    }
    fut().await;
}

fn assert_send(_: impl Send) {}

pub fn pass_assert() {
    assert_send(still_send());
}
