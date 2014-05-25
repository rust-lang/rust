// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn wants_uniq(x: String) { }
fn wants_slice(x: &str) { }

fn has_uniq(x: String) {
   wants_uniq(x);
   wants_slice(x.as_slice());
}

fn has_slice(x: &str) {
   wants_uniq(x); //~ ERROR mismatched types
   wants_slice(x);
}

fn main() {
}
