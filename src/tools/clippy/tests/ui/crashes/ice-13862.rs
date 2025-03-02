//@ check-pass

#![crate_type = "lib"]
#![no_std]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

pub struct S<const N: u8>;

impl<const N: u8> Future for S<N> {
    type Output = ();
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        todo!()
    }
}

pub fn f<const N: u8>() -> S<N> {
    S
}
