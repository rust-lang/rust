// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android: FIXME(#10381)

// aux-build:cross_crate_debuginfo_type_uniquing.rs
extern crate cross_crate_debuginfo_type_uniquing;

// no-prefer-dynamic
// compile-flags:-g -Zlto

pub struct C;
pub fn p() -> C {
    C
}

fn main() { }
