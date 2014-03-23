// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-stage1
// ignore-pretty
// ignore-cross-compile #12102

#![feature(phase)]

#[phase(syntax)]
extern crate hexfloat;

fn main() {
    hexfloat!("foo");
    //~^ ERROR invalid hex float literal in hexfloat!: Expected '0'
    hexfloat!("0");
    //~^ERROR invalid hex float literal in hexfloat!: Expected 'x'
    hexfloat!("0x");
    //~^ERROR invalid hex float literal in hexfloat!: Expected '.'
    hexfloat!("0x.");
    //~^ERROR invalid hex float literal in hexfloat!: Expected digits before or after decimal point
    hexfloat!("0x0.0");
    //~^ERROR invalid hex float literal in hexfloat!: Expected 'p'
    hexfloat!("0x0.0p");
    //~^ERROR invalid hex float literal in hexfloat!: Expected exponent digits
    hexfloat!("0x0.0p0f");
    //~^ERROR invalid hex float literal in hexfloat!: Expected end of string
}
