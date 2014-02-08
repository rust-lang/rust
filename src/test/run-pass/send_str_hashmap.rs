// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::clone::{Clone, DeepClone};
use std::cmp::{TotalEq, Ord, TotalOrd, Equiv};
use std::cmp::Equal;
use std::container::{Container, Map, MutableMap};
use std::default::Default;
use std::str::{Str, SendStr, Owned, Slice};
use std::to_str::ToStr;
use std::hashmap::HashMap;
use std::option::Some;

pub fn main() {
    let mut map: HashMap<SendStr, uint> = HashMap::new();
    assert!(map.insert(Slice("foo"), 42));
    assert!(!map.insert(Owned(~"foo"), 42));
    assert!(!map.insert(Slice("foo"), 42));
    assert!(!map.insert(Owned(~"foo"), 42));

    assert!(!map.insert(Slice("foo"), 43));
    assert!(!map.insert(Owned(~"foo"), 44));
    assert!(!map.insert(Slice("foo"), 45));
    assert!(!map.insert(Owned(~"foo"), 46));

    let v = 46;

    assert_eq!(map.find(&Owned(~"foo")), Some(&v));
    assert_eq!(map.find(&Slice("foo")), Some(&v));

    let (a, b, c, d) = (50, 51, 52, 53);

    assert!(map.insert(Slice("abc"), a));
    assert!(map.insert(Owned(~"bcd"), b));
    assert!(map.insert(Slice("cde"), c));
    assert!(map.insert(Owned(~"def"), d));

    assert!(!map.insert(Slice("abc"), a));
    assert!(!map.insert(Owned(~"bcd"), b));
    assert!(!map.insert(Slice("cde"), c));
    assert!(!map.insert(Owned(~"def"), d));

    assert!(!map.insert(Owned(~"abc"), a));
    assert!(!map.insert(Slice("bcd"), b));
    assert!(!map.insert(Owned(~"cde"), c));
    assert!(!map.insert(Slice("def"), d));

    assert_eq!(map.find_equiv(&("abc")), Some(&a));
    assert_eq!(map.find_equiv(&("bcd")), Some(&b));
    assert_eq!(map.find_equiv(&("cde")), Some(&c));
    assert_eq!(map.find_equiv(&("def")), Some(&d));

    assert_eq!(map.find_equiv(&(~"abc")), Some(&a));
    assert_eq!(map.find_equiv(&(~"bcd")), Some(&b));
    assert_eq!(map.find_equiv(&(~"cde")), Some(&c));
    assert_eq!(map.find_equiv(&(~"def")), Some(&d));

    assert_eq!(map.find_equiv(&Slice("abc")), Some(&a));
    assert_eq!(map.find_equiv(&Slice("bcd")), Some(&b));
    assert_eq!(map.find_equiv(&Slice("cde")), Some(&c));
    assert_eq!(map.find_equiv(&Slice("def")), Some(&d));

    assert_eq!(map.find_equiv(&Owned(~"abc")), Some(&a));
    assert_eq!(map.find_equiv(&Owned(~"bcd")), Some(&b));
    assert_eq!(map.find_equiv(&Owned(~"cde")), Some(&c));
    assert_eq!(map.find_equiv(&Owned(~"def")), Some(&d));
}
