// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![feature(fn_traits, unboxed_closures)]

#[allow(dead_code)]
struct Foo;

impl<'a> Fn<(&'a (),)> for Foo {
    extern "rust-call" fn call(&self, (_,): (&(),)) {}
}

impl<'a> FnMut<(&'a (),)> for Foo {
    extern "rust-call" fn call_mut(&mut self, (_,): (&(),)) {}
}

impl<'a> FnOnce<(&'a (),)> for Foo {
    type Output = ();

    extern "rust-call" fn call_once(self, (_,): (&(),)) {}
}

fn main() {}
