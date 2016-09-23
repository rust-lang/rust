// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(untagged_unions)]

macro_rules! duplicate {
   ($i: item) => {
        mod m1 {
            $i
        }
        mod m2 {
            $i
        }
   }
}

duplicate! {
    pub union U {
        pub a: u8
    }
}

fn main() {
    let u1 = m1::U { a: 0 };
    let u2 = m2::U { a: 0 };
}
