// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test default methods in PartialOrd and PartialEq
//
struct Fool(bool);

impl PartialEq for Fool {
    fn eq(&self, other: &Fool) -> bool {
        let Fool(this) = *self;
        let Fool(other) = *other;
        this != other
    }
}

struct Int(int);

impl PartialEq for Int {
    fn eq(&self, other: &Int) -> bool {
        let Int(this) = *self;
        let Int(other) = *other;
        this == other
    }
}

impl PartialOrd for Int {
    fn partial_cmp(&self, other: &Int) -> Option<Ordering> {
        let Int(this) = *self;
        let Int(other) = *other;
        this.partial_cmp(&other)
    }
}

struct RevInt(int);

impl PartialEq for RevInt {
    fn eq(&self, other: &RevInt) -> bool {
        let RevInt(this) = *self;
        let RevInt(other) = *other;
        this == other
    }
}

impl PartialOrd for RevInt {
    fn partial_cmp(&self, other: &RevInt) -> Option<Ordering> {
        let RevInt(this) = *self;
        let RevInt(other) = *other;
        other.partial_cmp(&this)
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

    assert!(Fool(true)  == Fool(false));
    assert!(Fool(true)  != Fool(true));
    assert!(Fool(false) != Fool(false));
    assert!(Fool(false) == Fool(true));
}
