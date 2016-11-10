// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(fn_traits, unboxed_closures)]

use std::ops::{Fn, FnMut, FnOnce};

struct SFn {
    x: isize,
    y: isize,
}

impl Fn<(isize,)> for SFn {
    extern "rust-call" fn call(&self, (z,): (isize,)) -> isize {
        self.x * self.y * z
    }
}

impl FnMut<(isize,)> for SFn {
    extern "rust-call" fn call_mut(&mut self, args: (isize,)) -> isize { self.call(args) }
}

impl FnOnce<(isize,)> for SFn {
    type Output = isize;
    extern "rust-call" fn call_once(self, args: (isize,)) -> isize { self.call(args) }
}

struct SFnMut {
    x: isize,
    y: isize,
}

impl FnMut<(isize,)> for SFnMut {
    extern "rust-call" fn call_mut(&mut self, (z,): (isize,)) -> isize {
        self.x * self.y * z
    }
}

impl FnOnce<(isize,)> for SFnMut {
    type Output = isize;
    extern "rust-call" fn call_once(mut self, args: (isize,)) -> isize { self.call_mut(args) }
}

struct SFnOnce {
    x: String,
}

impl FnOnce<(String,)> for SFnOnce {
    type Output = usize;

    extern "rust-call" fn call_once(self, (z,): (String,)) -> usize {
        self.x.len() + z.len()
    }
}

fn f() {
    let mut s = SFn {
        x: 1,
        y: 2,
    };
    let sp = &mut s;
    s(3);   //~ ERROR cannot borrow `s` as immutable because it is also borrowed as mutable
    //~^ ERROR cannot borrow `s` as immutable because it is also borrowed as mutable
}

fn g() {
    let s = SFnMut {
        x: 1,
        y: 2,
    };
    s(3);   //~ ERROR cannot borrow immutable local variable `s` as mutable
}

fn h() {
    let s = SFnOnce {
        x: "hello".to_string(),
    };
    s(" world".to_string());
    s(" world".to_string());    //~ ERROR use of moved value: `s`
}

fn main() {}
