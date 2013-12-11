// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that assignments to an `&mut` pointer which is found in a
// borrowed (but otherwise non-aliasable) location is illegal.

struct S<'a> {
    pointer: &'a mut int
}

fn a(s: &S) {
    *s.pointer += 1; //~ ERROR cannot assign
}

fn b(s: &mut S) {
    *s.pointer += 1;
}

fn c(s: & &mut S) {
    *s.pointer += 1; //~ ERROR cannot assign
}

fn main() {}
