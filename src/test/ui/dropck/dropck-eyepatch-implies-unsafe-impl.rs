// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generic_param_attrs)]
#![feature(dropck_eyepatch)]

// This test ensures that a use of `#[may_dangle]` is rejected if
// it is not attached to an `unsafe impl`.

use std::fmt;

struct Dt<A: fmt::Debug>(&'static str, A);
struct Dr<'a, B:'a+fmt::Debug>(&'static str, &'a B);
struct Pt<A,B: fmt::Debug>(&'static str, A, B);
struct Pr<'a, 'b, B:'a+'b+fmt::Debug>(&'static str, &'a B, &'b B);
struct St<A: fmt::Debug>(&'static str, A);
struct Sr<'a, B:'a+fmt::Debug>(&'static str, &'a B);

impl<A: fmt::Debug> Drop for Dt<A> {
    fn drop(&mut self) { println!("drop {} {:?}", self.0, self.1); }
}
impl<'a, B: fmt::Debug> Drop for Dr<'a, B> {
    fn drop(&mut self) { println!("drop {} {:?}", self.0, self.1); }
}
impl<#[may_dangle] A, B: fmt::Debug> Drop for Pt<A, B> {
    //~^ ERROR requires an `unsafe impl` declaration due to `#[may_dangle]` attribute

    // (unsafe to access self.1  due to #[may_dangle] on A)
    fn drop(&mut self) { println!("drop {} {:?}", self.0, self.2); }
}
impl<#[may_dangle] 'a, 'b, B: fmt::Debug> Drop for Pr<'a, 'b, B> {
    //~^ ERROR requires an `unsafe impl` declaration due to `#[may_dangle]` attribute

    // (unsafe to access self.1 due to #[may_dangle] on 'a)
    fn drop(&mut self) { println!("drop {} {:?}", self.0, self.2); }
}

fn main() {
}
