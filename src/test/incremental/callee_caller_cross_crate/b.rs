// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:a.rs
// revisions:rpass1 rpass2
// compile-flags:-Z query-dep-graph

#![feature(rustc_attrs)]

extern crate a;

#[rustc_dirty(label="TypeckItemBody", cfg="rpass2")]
pub fn call_function0() {
    a::function0(77);
}

#[rustc_clean(label="TypeckItemBody", cfg="rpass2")]
pub fn call_function1() {
    a::function1(77);
}

pub fn main() { }
