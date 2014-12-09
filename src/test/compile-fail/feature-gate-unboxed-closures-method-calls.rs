// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

fn foo<F: Fn<(), ()>>(mut f: F) {
    f.call(()); //~ ERROR explicit use of unboxed closure method `call`
    f.call_mut(()); //~ ERROR explicit use of unboxed closure method `call_mut`
    f.call_once(()); //~ ERROR explicit use of unboxed closure method `call_once`
}

fn main() {}
