// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test default methods in Ord and Eq
//
struct Fool(bool);

impl Eq for Fool {
    fn eq(&self, other: &Fool) -> bool {
        let Fool(this) = *self;
        let Fool(other) = *other;
        this != other
    }
}

struct Int(int);

impl Ord for Int {
    fn lt(&self, other: &Int) -> bool {
        let Int(this) = *self;
        let Int(other) = *other;
        this < other
    }
}

struct RevInt(int);

impl Ord for RevInt {
    fn lt(&self, other: &RevInt) -> bool {
        let RevInt(this) = *self;
        let RevInt(other) = *other;
        this > other
    }
}

pub fn main() {
    fail_unless!(Int(2) >  Int(1));
    fail_unless!(Int(2) >= Int(1));
    fail_unless!(Int(1) >= Int(1));
    fail_unless!(Int(1) <  Int(2));
    fail_unless!(Int(1) <= Int(2));
    fail_unless!(Int(1) <= Int(1));

    fail_unless!(RevInt(2) <  RevInt(1));
    fail_unless!(RevInt(2) <= RevInt(1));
    fail_unless!(RevInt(1) <= RevInt(1));
    fail_unless!(RevInt(1) >  RevInt(2));
    fail_unless!(RevInt(1) >= RevInt(2));
    fail_unless!(RevInt(1) >= RevInt(1));

    fail_unless!(Fool(true)  == Fool(false));
    fail_unless!(Fool(true)  != Fool(true));
    fail_unless!(Fool(false) != Fool(false));
    fail_unless!(Fool(false) == Fool(true));
}
