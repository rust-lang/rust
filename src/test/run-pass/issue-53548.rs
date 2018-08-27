// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #53548: having a 'static bound on a trait
// made it impossible to keep a trait object to it across an
// await point inside a closure

#![feature(arbitrary_self_types, async_await, await_macro, futures_api, pin)]

use std::future::Future;
use std::mem::PinMut;
use std::task::{Poll, Context};

// A trait with 'static bound
trait Trait: 'static {}

// Anything we can give to await!()
struct DummyFut();
impl Future for DummyFut {
    type Output = ();
    fn poll(self: PinMut<Self>, _ctx: &mut Context) -> Poll<()> {
        Poll::Pending
    }
}

// The actual reproducer, requires that Trait is 'static and a trait
// object to it is captured in a closure for successful reproduction.
async fn foo(b: Box<Trait + 'static>) -> () {
    let _bar = move || { b; () };
    await!(DummyFut())
}

pub fn main() {}
