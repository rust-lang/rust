// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:link-cfg-works-transitive-rlib.rs
// aux-build:link-cfg-works-transitive-dylib.rs

#![feature(link_cfg)]

extern crate link_cfg_works_transitive_rlib;
extern crate link_cfg_works_transitive_dylib;

#[link(name = "foo", cfg(foo))]
extern {}

fn main() {}

