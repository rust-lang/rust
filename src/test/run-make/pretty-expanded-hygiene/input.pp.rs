// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// minimal junk
#![feature(no_core)]
#![no_core]


fn bar /* 62#0 */() { let x /* 59#2 */ = 1; y /* 61#4 */ + x /* 59#5 */ }

fn y /* 61#0 */() { }
