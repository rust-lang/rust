// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure specialization cannot change impl polarity

#![feature(optin_builtin_traits)]
#![feature(specialization)]

trait Foo {}

#[allow(auto_impl)]
impl Foo for .. {}

impl<T> Foo for T {}
impl !Foo for u8 {} //~ ERROR E0119

trait Bar {}

#[allow(auto_impl)]
impl Bar for .. {}

impl<T> !Bar for T {}
impl Bar for u8 {} //~ ERROR E0119

fn main() {}
