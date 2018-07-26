// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(fn_traits)]

trait CallSingle<A, B> {
    fn call(&self, a: A) -> B where Self: Sized, Self: Fn(A) -> B;
}

impl<A, B, F: Fn(A) -> B> CallSingle<A, B> for F {
    fn call(&self, a: A) -> B {
        <Self as Fn(A) -> B>::call(self, (a,))
        //~^ ERROR associated type bindings are not allowed here
    }
}

fn main() {}
