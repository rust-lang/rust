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
static mut DEBUG_BORROW: bool = false;
static mut POISON_ON_FREE: bool = false;

pub fn init() {
    unsafe {
        match os::getenv("RUST_MIN_STACK") {
            Some(s) => match FromStr::from_str(s) {
                Some(i) => MIN_STACK = i,
                None => ()
            },
            None => ()
        }
        match os::getenv("RUST_DEBUG_BORROW") {
            Some(_) => DEBUG_BORROW = true,
            None => ()
        }
        match os::getenv("RUST_POISON_ON_FREE") {
            Some(_) => POISON_ON_FREE = true,
            None => ()
        }
    }
}

pub fn min_stack() -> uint {
    unsafe { MIN_STACK }
}

pub fn debug_borrow() -> bool {
    unsafe { DEBUG_BORROW }
}

pub fn poison_on_free() -> bool {
    unsafe { POISON_ON_FREE }
}
