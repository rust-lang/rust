// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


struct F { f: Vec<int> }

fn impure(_v: &[int]) {
}

pub fn main() {
    let mut x = F {f: vec!(3)};

    match x {
      F {f: ref mut v} => {
        impure(v.as_slice());
      }
    }
}
