// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unsafe_destructor)]

struct defer<'a> {
    x: &'a [&'a str],
}

#[unsafe_destructor]
impl<'a> Drop for defer<'a> {
    fn drop(&mut self) {
        unsafe {
            println!("{}", self.x);
        }
    }
}

fn defer<'r>(x: &'r [&'r str]) -> defer<'r> {
    defer {
        x: x
    }
}

fn main() {
    let x = defer(vec!("Goodbye", "world!").as_slice());
    //~^ ERROR borrowed value does not live long enough
    x.x[0];
}
