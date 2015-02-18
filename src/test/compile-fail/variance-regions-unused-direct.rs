// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that disallow lifetime parameters that are unused.

use std::marker;

struct Bivariant<'a>; //~ ERROR parameter `'a` is never used

struct Struct<'a, 'd> { //~ ERROR parameter `'d` is never used
    field: &'a [i32]
}

trait Trait<'a, 'd> { //~ ERROR parameter `'d` is never used
    fn method(&'a self);
}

fn main() {}
