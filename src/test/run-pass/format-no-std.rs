// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten missing rust_begin_unwind

#![feature(lang_items, start, collections)]
#![no_std]

extern crate std as other;

#[macro_use] extern crate collections;

use collections::string::ToString;

#[start]
fn start(_argc: isize, _argv: *const *const u8) -> isize {
    let s = format!("{}", 1_isize);
    assert_eq!(s, "1".to_string());

    let s = format!("test");
    assert_eq!(s, "test".to_string());

    let s = format!("{test}", test=3_isize);
    assert_eq!(s, "3".to_string());

    let s = format!("hello {}", "world");
    assert_eq!(s, "hello world".to_string());

    0
}
