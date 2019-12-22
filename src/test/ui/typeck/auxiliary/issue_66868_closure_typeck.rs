// edition:2018

#![crate_type = "lib"]

use std::{
    future::Future,
    pin::Pin,
    sync::RwLock,
    task::{Context, Poll},
};

struct S {}

impl Future for S {
    type Output = ();
    fn poll(self: Pin<&mut Self>, _: &mut Context) -> Poll<Self::Output> {
        Poll::Pending
    }
}

pub async fn f() {
    let fo = RwLock::new(S {});

    (&mut *fo.write().unwrap()).await;
}
