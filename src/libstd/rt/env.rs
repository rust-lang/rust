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

use from_str::from_str;
use option::{Some, None};
use os;
use str::Str;

// Note that these are all accessed without any synchronization.
// They are expected to be initialized once then left alone.

static mut MIN_STACK: uint = 2 * 1024 * 1024;
/// This default corresponds to 20M of cache per scheduler (at the default size).
static mut MAX_CACHED_STACKS: uint = 10;
static mut DEBUG_BORROW: bool = false;

pub fn init() {
    unsafe {
        match os::getenv("RUST_MIN_STACK") {
            Some(s) => match from_str(s.as_slice()) {
                Some(i) => MIN_STACK = i,
                None => ()
            },
            None => ()
        }
        match os::getenv("RUST_MAX_CACHED_STACKS") {
            Some(max) => {
                MAX_CACHED_STACKS =
                    from_str(max.as_slice()).expect("expected positive \
                                                     integer in \
                                                     RUST_MAX_CACHED_STACKS")
            }
            None => ()
        }
        match os::getenv("RUST_DEBUG_BORROW") {
            Some(_) => DEBUG_BORROW = true,
            None => ()
        }
    }
}

pub fn min_stack() -> uint {
    unsafe { MIN_STACK }
}

pub fn max_cached_stacks() -> uint {
    unsafe { MAX_CACHED_STACKS }
}

pub fn debug_borrow() -> bool {
    unsafe { DEBUG_BORROW }
}
