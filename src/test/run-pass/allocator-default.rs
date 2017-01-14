// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(alloc_jemalloc)]

#[cfg(any(all(target_os = "linux", any(target_arch = "x86", target_arch = "x86_64")),
          target_os = "macos"))]
extern crate alloc_jemalloc;

fn main() {
    println!("{:?}", Box::new(3));
}
