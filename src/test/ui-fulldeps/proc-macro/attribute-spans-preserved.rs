// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:attribute-spans-preserved.rs

extern crate attribute_spans_preserved as foo;

use foo::foo;

#[ foo ( let y: u32 = "z"; ) ] //~ ERROR: mismatched types
#[ bar { let x: u32 = "y"; } ] //~ ERROR: mismatched types
fn main() {
}
