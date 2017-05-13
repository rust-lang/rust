// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

fn needs_fn<F>(x: F) where F: Fn(isize) -> isize {}

fn main() {
    let _: () = (box |_: isize| {}) as Box<FnOnce(isize)>;
    //~^ ERROR mismatched types
    //~| expected type `()`
    //~| found type `Box<std::ops::FnOnce(isize)>`
    //~| expected (), found box
    let _: () = (box |_: isize, isize| {}) as Box<Fn(isize, isize)>;
    //~^ ERROR mismatched types
    //~| expected type `()`
    //~| found type `Box<std::ops::Fn(isize, isize)>`
    //~| expected (), found box
    let _: () = (box || -> isize { unimplemented!() }) as Box<FnMut() -> isize>;
    //~^ ERROR mismatched types
    //~| expected type `()`
    //~| found type `Box<std::ops::FnMut() -> isize>`
    //~| expected (), found box

    needs_fn(1);
    //~^ ERROR : std::ops::Fn<(isize,)>`
    //~| ERROR : std::ops::FnOnce<(isize,)>`
}
