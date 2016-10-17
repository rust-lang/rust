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

// This is a support file for ../dropck-eyepatch-extern-crate.rs
//
// The point of this test is to illustrate that the `#[may_dangle]`
// attribute specifically allows, in the context of a type
// implementing `Drop`, a generic parameter to be instantiated with a
// lifetime that does not strictly outlive the owning type itself,
// and that this attribute's effects are preserved when importing
// the type from another crate.
//
// See also ../dropck-eyepatch.rs for more information about the general
// structure of the test.

use std::fmt;

pub struct Dt<A: fmt::Debug>(pub &'static str, pub A);
pub struct Dr<'a, B:'a+fmt::Debug>(pub &'static str, pub &'a B);
pub struct Pt<A,B: fmt::Debug>(pub &'static str, pub A, pub B);
pub struct Pr<'a, 'b, B:'a+'b+fmt::Debug>(pub &'static str, pub &'a B, pub &'b B);
pub struct St<A: fmt::Debug>(pub &'static str, pub A);
pub struct Sr<'a, B:'a+fmt::Debug>(pub &'static str, pub &'a B);

impl<A: fmt::Debug> Drop for Dt<A> {
    fn drop(&mut self) { println!("drop {} {:?}", self.0, self.1); }
}
impl<'a, B: fmt::Debug> Drop for Dr<'a, B> {
    fn drop(&mut self) { println!("drop {} {:?}", self.0, self.1); }
}
unsafe impl<#[may_dangle] A, B: fmt::Debug> Drop for Pt<A, B> {
    // (unsafe to access self.1  due to #[may_dangle] on A)
    fn drop(&mut self) { println!("drop {} {:?}", self.0, self.2); }
}
unsafe impl<#[may_dangle] 'a, 'b, B: fmt::Debug> Drop for Pr<'a, 'b, B> {
    // (unsafe to access self.1 due to #[may_dangle] on 'a)
    fn drop(&mut self) { println!("drop {} {:?}", self.0, self.2); }
}
