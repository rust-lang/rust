// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::marker::Pod;

struct Atom {
    data: u64,
}

impl Drop for Atom {
    fn drop(&mut self) {
        // decrease refcount
    }
}

impl Pod for Atom {}
//~^ error: the trait `Pod` may not be implemented for this type; the type has a destructor

fn main() {}
