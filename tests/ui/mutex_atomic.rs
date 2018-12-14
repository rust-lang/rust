// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::all)]
#![warn(clippy::mutex_integer)]

fn main() {
    use std::sync::Mutex;
    Mutex::new(true);
    Mutex::new(5usize);
    Mutex::new(9isize);
    let mut x = 4u32;
    Mutex::new(&x as *const u32);
    Mutex::new(&mut x as *mut u32);
    Mutex::new(0u32);
    Mutex::new(0i32);
    Mutex::new(0f32); // there are no float atomics, so this should not lint
}
