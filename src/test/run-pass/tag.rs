// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




// -*- rust -*-
enum colour { red(int, int), green, }

impl cmp::Eq for colour {
    fn eq(&self, other: &colour) -> bool {
        match *self {
            red(a0, b0) => {
                match (*other) {
                    red(a1, b1) => a0 == a1 && b0 == b1,
                    green => false,
                }
            }
            green => {
                match (*other) {
                    red(*) => false,
                    green => true
                }
            }
        }
    }
    fn ne(&self, other: &colour) -> bool { !(*self).eq(other) }
}

fn f() { let x = red(1, 2); let y = green; assert!((x != y)); }

pub fn main() { f(); }
