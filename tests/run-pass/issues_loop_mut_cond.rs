// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

/// Issue: https://github.com/rust-lang/rust-clippy/issues/2596
pub fn loop_on_block_condition(u: &mut isize) {
    while { *u < 0 } {
        *u += 1;
    }
}

/// https://github.com/rust-lang/rust-clippy/issues/2584
fn loop_with_unsafe_condition(ptr: *const u8) {
    let mut len = 0;
    while unsafe { *ptr.offset(len) } != 0 {
        len += 1;
    }
}

/// https://github.com/rust-lang/rust-clippy/issues/2710
static mut RUNNING: bool = true;
fn loop_on_static_condition() {
    unsafe {
        while RUNNING {
            RUNNING = false;
        }
    }
}

fn main() {}
