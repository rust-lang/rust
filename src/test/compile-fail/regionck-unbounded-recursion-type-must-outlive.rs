// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Prevent unbounded recursion during a type_must_outlive pass
// in regionck.

struct B<F: Fn()> (F);

fn main() {
    let mut p = B(std::mem::zeroed());
    p.0 = ||{p.0;} ;
}
