// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_private)]
#![feature(staged_api)]
#![cfg_attr(all(feature = "rustc_alloc_frame", stage1), feature(alloc_frame))]

extern crate rustc_driver;

// Use the frame allocator to speed up runtime
#[cfg(all(feature = "rustc_alloc_frame", stage1))]
extern crate alloc_frame;

fn main() { rustc_driver::main() }
