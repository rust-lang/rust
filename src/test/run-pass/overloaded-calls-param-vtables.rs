// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that nested vtables work with overloaded calls.

// pretty-expanded FIXME #23616

#![feature(unboxed_closures, fn_traits)]

use std::marker::PhantomData;
use std::ops::Fn;
use std::ops::Add;

struct G<A>(PhantomData<A>);

impl<'a, A: Add<i32, Output=i32>> Fn<(A,)> for G<A> {
    extern "rust-call" fn call(&self, (arg,): (A,)) -> i32 {
        arg.add(1)
    }
}

impl<'a, A: Add<i32, Output=i32>> FnMut<(A,)> for G<A> {
    extern "rust-call" fn call_mut(&mut self, args: (A,)) -> i32 { self.call(args) }
}

impl<'a, A: Add<i32, Output=i32>> FnOnce<(A,)> for G<A> {
    type Output = i32;
    extern "rust-call" fn call_once(self, args: (A,)) -> i32 { self.call(args) }
}

fn main() {
    // ICE trigger
    (G(PhantomData))(1);
}
