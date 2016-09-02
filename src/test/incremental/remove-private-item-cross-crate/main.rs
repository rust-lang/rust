// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we are able to reuse `main` even though a private
// item was removed from the root module of crate`a`.

// revisions:rpass1 rpass2
// aux-build:a.rs

#![feature(rustc_attrs)]
#![crate_type = "bin"]
#![rustc_partition_reused(module="main", cfg="rpass2")]

extern crate a;

pub fn main() {
    let vec: Vec<u8> = vec![0, 1, 2, 3];
    for &b in &vec {
        println!("{}", a::foo(b));
    }
}
