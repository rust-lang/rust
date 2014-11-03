// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that object-safe methods are identified as such.  Also
// acts as a regression test for #18490

trait Tr {
    // Static methods are always safe regardless of other rules
    fn new() -> Self;
}

struct St;

impl Tr for St {
    fn new() -> St { St }
}

fn main() {
    &St as &Tr;
}
