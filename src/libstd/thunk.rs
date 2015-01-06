// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Because this module is temporary...
#![allow(missing_docs)]

use alloc::boxed::Box;
use core::marker::Send;
use core::ops::FnOnce;

pub struct Thunk<A=(),R=()> {
    invoke: Box<Invoke<A,R>+Send>
}

impl<R> Thunk<(),R> {
    pub fn new<F>(func: F) -> Thunk<(),R>
        where F : FnOnce() -> R, F : Send
    {
        Thunk::with_arg(move|: ()| func())
    }
}

impl<A,R> Thunk<A,R> {
    pub fn with_arg<F>(func: F) -> Thunk<A,R>
        where F : FnOnce(A) -> R, F : Send
    {
        Thunk {
            invoke: box func
        }
    }

    pub fn invoke(self, arg: A) -> R {
        self.invoke.invoke(arg)
    }
}

pub trait Invoke<A=(),R=()> {
    fn invoke(self: Box<Self>, arg: A) -> R;
}

impl<A,R,F> Invoke<A,R> for F
    where F : FnOnce(A) -> R
{
    fn invoke(self: Box<F>, arg: A) -> R {
        let f = *self;
        f(arg)
    }
}
