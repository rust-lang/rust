// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "rlib"]
struct Foo;

trait Tr {
    fn tr(&self);
}

impl Tr for Foo {
    fn tr(&self) {}
}

fn take_method<T>(f: fn(&T), t: &T) {}

#[inline]
pub fn pass_method() {
    take_method(Tr::tr, &Foo);
}
