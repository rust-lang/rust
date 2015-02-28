// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![forbid(large_const_items)]
#![allow(dead_code)]

const FOO: [u8; 100] = [0; 100]; //~ ERROR using large `const` items (100 bytes) is not recommended

const BAR: &'static [u8; 100] = &[0; 100];
const BAZ: &'static str = "Hello!";

fn main() { }
