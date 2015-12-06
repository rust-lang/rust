// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that param substitutions from the correct environment are
// used when translating unboxed closure calls.

// pretty-expanded FIXME #23616

pub fn inside<F: Fn()>(c: F) {
    c();
}

// Use different number of type parameters and closure type to trigger
// an obvious ICE when param environments are mixed up
pub fn outside<A,B>() {
    inside(|| {});
}

fn main() {
    outside::<(),()>();
}
