// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test verifies that temporary lifetime is correctly computed
// for static objects in enclosing scopes.

extern crate extra;
use std::cmp::Eq;

fn f<T:Eq>(o: &mut Option<T>) {
    assert!(*o == None);
}

pub fn main() {
    mod t {
        enum E {V=1, A=0}
        static C: E = V;
    }

    f::<int>(&mut None);
}
