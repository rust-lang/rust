// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum E { V1(int), V0 }
static C: &'static [E] = &[V0, V1(0xDEADBEE), V0];

pub fn main() {
    match C[1] {
        V1(n) => assert!(n == 0xDEADBEE),
        _ => fail!()
    }
    match C[2] { 
        V0 => (),
        _ => fail!()
    }
}
