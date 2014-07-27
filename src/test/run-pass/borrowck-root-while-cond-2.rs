// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::gc::{GC, Gc};

struct F { f: Gc<G> }
struct G { g: Vec<int> }

pub fn main() {
    let rec = box(GC) F {f: box(GC) G {g: vec!(1, 2, 3)}};
    while rec.f.g.len() == 23 {}
}
