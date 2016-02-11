// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// DON'T REENABLE THIS UNLESS YOU'VE ACTUALLY FIXED THE UNDERLYING ISSUE
// ignore-android seems to block forever

// ignore-emscripten no threads support

#![forbid(warnings)]

// Pretty printing tests complain about `use std::predule::*`
#![allow(unused_imports)]

// A var moved into a proc, that has a mutable loan path should
// not trigger a misleading unused_mut warning.

use std::io::prelude::*;
use std::thread;

pub fn main() {
    let mut stdin = std::io::stdin();
    thread::spawn(move|| {
        let mut v = Vec::new();
        let _ = stdin.read_to_end(&mut v);
    }).join().ok().unwrap();
}
