// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test default methods in Ord
//
struct Int(int);

impl Ord for Int {
    fn lt(&self, other: &Int) -> bool {
        **self < **other
    }
}

struct RevInt(int);

impl Ord for RevInt {
    fn lt(&self, other: &RevInt) -> bool {
        **self > **other
    }
}

pub fn main() {
    assert!(Int(2) >  Int(1));
    assert!(Int(2) >= Int(1));
    assert!(Int(1) >= Int(1));
    assert!(Int(1) <  Int(2));
    assert!(Int(1) <= Int(2));
    assert!(Int(1) <= Int(1));

    assert!(RevInt(2) <  RevInt(1));
    assert!(RevInt(2) <= RevInt(1));
    assert!(RevInt(1) <= RevInt(1));
    assert!(RevInt(1) >  RevInt(2));
    assert!(RevInt(1) >= RevInt(2));
    assert!(RevInt(1) >= RevInt(1));
}
