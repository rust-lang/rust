// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic
// aux-build:cgu_test.rs
// aux-build:cgu_test_a.rs
// aux-build:cgu_test_b.rs

extern crate cgu_test_a;
extern crate cgu_test_b;

fn main() {
    cgu_test_a::a::a();
    cgu_test_b::a::a();
}
