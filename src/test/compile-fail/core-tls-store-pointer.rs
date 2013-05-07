// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Testing that we can't store a borrowed pointer it task-local storage

use core::task::local_data::*;

fn key(_x: @&int) { }

fn main() {
    unsafe {
        local_data_set(key, @&0); //~ ERROR does not fulfill `'static`
    }
}