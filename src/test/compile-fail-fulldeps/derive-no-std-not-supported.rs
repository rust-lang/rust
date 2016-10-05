// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![no_std]

extern crate rand;
extern crate serialize as rustc_serialize;

#[derive(RustcEncodable)]  //~ ERROR this trait cannot be derived
struct Bar {
    x: u32,
}

#[derive(RustcDecodable)]  //~ ERROR this trait cannot be derived
struct Baz {
    x: u32,
}

fn main() {
    Foo { x: 0 };
    Bar { x: 0 };
    Baz { x: 0 };
}
