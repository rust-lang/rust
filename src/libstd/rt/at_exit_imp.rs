// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of running at_exit routines
//!
//! Documentation can be found on the `rt::at_exit` function.

use cast;
use option::{Some, None};
use ptr::RawPtr;
use unstable::sync::Exclusive;
use util;

type Queue = Exclusive<~[proc()]>;

static mut QUEUE: *mut Queue = 0 as *mut Queue;
static mut RUNNING: bool = false;

pub fn init() {
    unsafe {
        rtassert!(!RUNNING);
        rtassert!(QUEUE.is_null());
        let state: ~Queue = ~Exclusive::new(~[]);
        QUEUE = cast::transmute(state);
    }
}

pub fn push(f: proc()) {
    unsafe {
        rtassert!(!RUNNING);
        rtassert!(!QUEUE.is_null());
        let state: &mut Queue = cast::transmute(QUEUE);
        let mut f = Some(f);
        state.with(|arr|  {
            arr.push(f.take_unwrap());
        });
    }
}

pub fn run() {
    let vec = unsafe {
        rtassert!(!RUNNING);
        rtassert!(!QUEUE.is_null());
        RUNNING = true;
        let state: ~Queue = cast::transmute(QUEUE);
        QUEUE = 0 as *mut Queue;
        let mut vec = None;
        state.with(|arr| {
            vec = Some(util::replace(arr, ~[]));
        });
        vec.take_unwrap()
    };


    for f in vec.move_iter() {
        f();
    }
}
