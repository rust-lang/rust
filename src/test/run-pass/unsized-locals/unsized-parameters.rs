// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unsized_locals)]

pub fn f0(_f: dyn FnOnce()) {}
pub fn f1(_s: str) {}
pub fn f2((_x, _y): (i32, [i32])) {}

fn main() {
    let foo = "foo".to_string().into_boxed_str();
    f1(*foo);
}
