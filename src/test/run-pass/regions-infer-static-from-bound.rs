// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that a 'static bound influences the lifetime we infer for a borrow.
// This test should compile.

static i: uint = 3;
fn foo1<T:Send>(t: T) { }
fn foo2<T:'static>(t: T) { }
pub fn main() {
    foo1(&i);
    foo2(&i);
}
