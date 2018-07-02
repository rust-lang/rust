// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:call-site.rs
// ignore-stage1

#![feature(proc_macro, proc_macro_non_items)]

extern crate call_site;
use call_site::*;

fn main() {
    let x1 = 10;
    call_site::check!(let x2 = x1;);
    let x6 = x5;
}
