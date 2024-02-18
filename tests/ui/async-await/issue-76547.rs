// Test for diagnostic improvement issue #76547
//@ edition:2018

use std::{
    future::Future,
    task::{Context, Poll}
};
use std::pin::Pin;

pub struct ListFut<'a>(&'a mut [&'a mut [u8]]);
impl<'a> Future for ListFut<'a> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<Self::Output> {
        unimplemented!()
    }
}

async fn fut(bufs: &mut [&mut [u8]]) {
    ListFut(bufs).await
    //~^ ERROR lifetime may not live long enough
}

pub struct ListFut2<'a>(&'a mut [&'a mut [u8]]);
impl<'a> Future for ListFut2<'a> {
    type Output = i32;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<Self::Output> {
        unimplemented!()
    }
}

async fn fut2(bufs: &mut [&mut [u8]]) -> i32 {
    ListFut2(bufs).await
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
