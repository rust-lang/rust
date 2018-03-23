// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]
#![deny(warnings)]

use std::env;
use std::thread;

fn main() {
    let should_fail = env::args().nth(1) == Some("bad".to_string());

    assert_eq!(thread::spawn(debug_assert_eq).join().is_err(), should_fail);
    assert_eq!(thread::spawn(debug_assert).join().is_err(), should_fail);
    assert_eq!(thread::spawn(overflow).join().is_err(), should_fail);
}

fn debug_assert_eq() {
    let mut hit1 = false;
    let mut hit2 = false;
    debug_assert_eq!({ hit1 = true; 1 }, { hit2 = true; 2 });
    assert!(!hit1);
    assert!(!hit2);
}

fn debug_assert() {
    let mut hit = false;
    debug_assert!({ hit = true; false });
    assert!(!hit);
}

fn overflow() {
    fn add(a: u8, b: u8) -> u8 { a + b }

    add(200u8, 200u8);
}
