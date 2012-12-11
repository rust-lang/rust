// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct yes0 {
  x: &uint,
}

impl yes0 : Drop {
    fn finalize(&self) {}
}

struct yes1 {
  x: &self/uint,
}

impl yes1 : Drop {
    fn finalize(&self) {}
}

struct yes2 {
  x: &foo/uint, //~ ERROR named regions other than `self` are not allowed as part of a type declaration
}

impl yes2 : Drop {
    fn finalize(&self) {}
}

fn main() {}
