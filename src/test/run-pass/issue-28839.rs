// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty : (#23623) problems with newlines before // comments

pub struct Foo;

pub fn get_foo2<'a>(foo: &'a mut Option<&'a mut Foo>) -> &'a mut Foo {
    match foo {
        // Ensure that this is not considered a move, but rather a reborrow.
        &mut Some(ref mut x) => *x,
        &mut None => panic!(),
    }
}

fn main() {
}
