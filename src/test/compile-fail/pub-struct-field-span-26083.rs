// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #26083
// Test that span for public struct fields start at `pub` instead of the identifier

struct Foo {
    pub bar: u8,

    pub
    //~^ error: field `bar` is already declared [E0124]
    bar: u8,

    pub bar:
    //~^ error: field `bar` is already declared [E0124]
    u8,

    bar:
    //~^ error: field `bar` is already declared [E0124]
    u8,
}

fn main() { }
