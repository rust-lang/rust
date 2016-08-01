// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten no threads support

use std::thread;
use std::mem;

fn main() {
    let y = 0u8;
    let closure = move |x: u8| y + x;

    // Check that both closures are capturing by value
    assert_eq!(1, mem::size_of_val(&closure));

    thread::spawn(move|| {
        let ok = closure;
    }).join().ok().unwrap();
}
