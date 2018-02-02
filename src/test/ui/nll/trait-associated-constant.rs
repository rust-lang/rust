// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test cases where we put various lifetime constraints on trait
// associated constants.

#![feature(rustc_attrs)]

use std::option::Option;

trait Anything<'a: 'b, 'b> {
    const AC: Option<&'b str>;
}

struct OKStruct { }

impl<'a: 'b, 'b> Anything<'a, 'b> for OKStruct {
    const AC: Option<&'b str> = None;
}

struct FailStruct1 { }

impl<'a: 'b, 'b, 'c> Anything<'a, 'b> for FailStruct1 {
    const AC: Option<&'c str> = None;
    //~^ ERROR: mismatched types
}

struct FailStruct2 { }

impl<'a: 'b, 'b> Anything<'a, 'b> for FailStruct2 {
    const AC: Option<&'a str> = None;
    //~^ ERROR: mismatched types
}

fn main() {}
