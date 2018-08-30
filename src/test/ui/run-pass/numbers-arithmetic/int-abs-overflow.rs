// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z force-overflow-checks=on
// ignore-emscripten no threads support

use std::thread;

fn main() {
    assert!(thread::spawn(|| i8::min_value().abs()).join().is_err());
    assert!(thread::spawn(|| i16::min_value().abs()).join().is_err());
    assert!(thread::spawn(|| i32::min_value().abs()).join().is_err());
    assert!(thread::spawn(|| i64::min_value().abs()).join().is_err());
    assert!(thread::spawn(|| isize::min_value().abs()).join().is_err());
}
