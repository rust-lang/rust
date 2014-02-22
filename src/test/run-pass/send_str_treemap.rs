// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate collections;

use std::clone::{Clone, DeepClone};
use std::cmp::{TotalEq, Ord, TotalOrd, Equiv};
use std::cmp::Equal;
use std::container::{Container, Map, MutableMap};
use std::default::Default;
use std::str::{Str, SendStr, Owned, Slice};
use std::to_str::ToStr;
use self::collections::TreeMap;
use std::option::Some;

pub fn main() {
    let mut map: TreeMap<SendStr, uint> = TreeMap::new();
    fail_unless!(map.insert(Slice("foo"), 42));
    fail_unless!(!map.insert(Owned(~"foo"), 42));
    fail_unless!(!map.insert(Slice("foo"), 42));
    fail_unless!(!map.insert(Owned(~"foo"), 42));

    fail_unless!(!map.insert(Slice("foo"), 43));
    fail_unless!(!map.insert(Owned(~"foo"), 44));
    fail_unless!(!map.insert(Slice("foo"), 45));
    fail_unless!(!map.insert(Owned(~"foo"), 46));

    let v = 46;

    fail_unless_eq!(map.find(&Owned(~"foo")), Some(&v));
    fail_unless_eq!(map.find(&Slice("foo")), Some(&v));

    let (a, b, c, d) = (50, 51, 52, 53);

    fail_unless!(map.insert(Slice("abc"), a));
    fail_unless!(map.insert(Owned(~"bcd"), b));
    fail_unless!(map.insert(Slice("cde"), c));
    fail_unless!(map.insert(Owned(~"def"), d));

    fail_unless!(!map.insert(Slice("abc"), a));
    fail_unless!(!map.insert(Owned(~"bcd"), b));
    fail_unless!(!map.insert(Slice("cde"), c));
    fail_unless!(!map.insert(Owned(~"def"), d));

    fail_unless!(!map.insert(Owned(~"abc"), a));
    fail_unless!(!map.insert(Slice("bcd"), b));
    fail_unless!(!map.insert(Owned(~"cde"), c));
    fail_unless!(!map.insert(Slice("def"), d));

    fail_unless_eq!(map.find(&Slice("abc")), Some(&a));
    fail_unless_eq!(map.find(&Slice("bcd")), Some(&b));
    fail_unless_eq!(map.find(&Slice("cde")), Some(&c));
    fail_unless_eq!(map.find(&Slice("def")), Some(&d));

    fail_unless_eq!(map.find(&Owned(~"abc")), Some(&a));
    fail_unless_eq!(map.find(&Owned(~"bcd")), Some(&b));
    fail_unless_eq!(map.find(&Owned(~"cde")), Some(&c));
    fail_unless_eq!(map.find(&Owned(~"def")), Some(&d));

    fail_unless!(map.pop(&Slice("foo")).is_some());
    fail_unless_eq!(map.move_iter().map(|(k, v)| k.to_str() + v.to_str())
                              .to_owned_vec()
                              .concat(),
               ~"abc50bcd51cde52def53");
}
