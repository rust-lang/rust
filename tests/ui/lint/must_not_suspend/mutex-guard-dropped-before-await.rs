//! Regression test for <https://github.com/rust-lang/rust/issues/89562>

//@ edition:2018
//@ run-pass

#![feature(must_not_suspend)]
#![deny(must_not_suspend)]

use std::sync::Mutex;

pub async fn foo() {
    let foo = Mutex::new(1);
    let lock = foo.lock().unwrap();

    // Prevent mutex lock being held across `.await` point.
    drop(lock);

    bar().await;
}

async fn bar() {}

fn main() {}
