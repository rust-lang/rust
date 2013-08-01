// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Runtime environment settings

use from_str::FromStr;
use option::{Some, None};
use os;

// Note that these are all accessed without any synchronization.
// They are expected to be initialized once then left alone.

static mut MIN_STACK: uint = 2000000;

pub fn init() {
    unsafe {
        match os::getenv("RUST_MIN_STACK") {
            Some(s) => match FromStr::from_str(s) {
                Some(i) => MIN_STACK = i,
                None => ()
            },
            None => ()
        }
    }
}

pub fn min_stack() -> uint {
    unsafe { MIN_STACK }
}
