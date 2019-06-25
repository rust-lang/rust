#![deny(unused_must_use)]
#![feature(arbitrary_self_types)]

use std::iter::Iterator;
use std::future::Future;

use std::task::{Context, Poll};
use std::pin::Pin;
use std::unimplemented;

struct MyFuture;

impl Future for MyFuture {
   type Output = u32;

   fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<u32> {
      Poll::Pending
   }
}

fn iterator() -> impl Iterator {
   std::iter::empty::<u32>()
}

fn future() -> impl Future {
   MyFuture
}

fn square_fn_once() -> impl FnOnce(u32) -> u32 {
   |x| x * x
}

fn square_fn_mut() -> impl FnMut(u32) -> u32 {
   |x| x * x
}

fn square_fn() -> impl Fn(u32) -> u32 {
   |x| x * x
}

fn main() {
   iterator(); //~ ERROR unused implementer of `std::iter::Iterator` that must be used
   future(); //~ ERROR unused implementer of `std::future::Future` that must be used
   square_fn_once(); //~ ERROR unused implementer of `std::ops::FnOnce` that must be used
   square_fn_mut(); //~ ERROR unused implementer of `std::ops::FnMut` that must be used
   square_fn(); //~ ERROR unused implementer of `std::ops::Fn` that must be used
}
