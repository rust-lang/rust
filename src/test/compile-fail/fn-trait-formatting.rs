// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unboxed_closures)]
#![feature(box_syntax)]

fn needs_fn<F>(x: F) where F: Fn(isize) -> isize {}

fn main() {
    let _: () = (box |:_: isize| {}) as Box<FnOnce(isize)>;
    //~^ ERROR object-safe
    //~| ERROR mismatched types
    //~| expected `()`
    //~| found `Box<core::ops::FnOnce(isize)>`
    //~| expected ()
    //~| found box
    let _: () = (box |&:_: isize, isize| {}) as Box<Fn(isize, isize)>;
    //~^ ERROR mismatched types
    //~| expected `()`
    //~| found `Box<core::ops::Fn(isize, isize)>`
    //~| expected ()
    //~| found box
    let _: () = (box |&mut:| -> isize unimplemented!()) as Box<FnMut() -> isize>;
    //~^ ERROR mismatched types
    //~| expected `()`
    //~| found `Box<core::ops::FnMut() -> isize>`
    //~| expected ()
    //~| found box

    needs_fn(1is);
    //~^ ERROR `core::ops::Fn<(isize,)>`
    //~| ERROR `core::ops::Fn<(isize,)>`
}
