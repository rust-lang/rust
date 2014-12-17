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

fn needs_fn<F>(x: F) where F: Fn(int) -> int {}

fn main() {
    let _: () = (box |:_: int| {}) as Box<FnOnce(int)>; //~ ERROR object-safe
    //~^ ERROR Box<core::ops::FnOnce(int)>
    let _: () = (box |&:_: int, int| {}) as Box<Fn(int, int)>;
    //~^ ERROR Box<core::ops::Fn(int, int)>
    let _: () = (box |&mut:| -> int unimplemented!()) as Box<FnMut() -> int>;
    //~^ ERROR Box<core::ops::FnMut() -> int>

    needs_fn(1i); //~ ERROR `core::ops::Fn(int) -> int`
}
