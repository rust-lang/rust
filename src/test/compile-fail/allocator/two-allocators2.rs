// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:system-allocator.rs
// no-prefer-dynamic
// error-pattern: the #[global_allocator] in

#![feature(global_allocator, allocator_api)]

extern crate system_allocator;

use std::heap::System;

#[global_allocator]
static A: System = System;

fn main() {}

