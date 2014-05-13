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

use std::container::{ Map, MutableMap};
use std::str::{SendStr, Owned, Slice};
use std::to_str::ToStr;
use self::collections::TreeMap;
use std::option::Some;

pub fn main() {
    let mut map: TreeMap<SendStr, uint> = TreeMap::new();
    assert!(map.insert(Slice("foo"), 42));
    assert!(!map.insert(Owned("foo".to_owned()), 42));
    assert!(!map.insert(Slice("foo"), 42));
    assert!(!map.insert(Owned("foo".to_owned()), 42));

    assert!(!map.insert(Slice("foo"), 43));
    assert!(!map.insert(Owned("foo".to_owned()), 44));
    assert!(!map.insert(Slice("foo"), 45));
    assert!(!map.insert(Owned("foo".to_owned()), 46));

    let v = 46;

    assert_eq!(map.find(&Owned("foo".to_owned())), Some(&v));
    assert_eq!(map.find(&Slice("foo")), Some(&v));

    let (a, b, c, d) = (50, 51, 52, 53);

    assert!(map.insert(Slice("abc"), a));
    assert!(map.insert(Owned("bcd".to_owned()), b));
    assert!(map.insert(Slice("cde"), c));
    assert!(map.insert(Owned("def".to_owned()), d));

    assert!(!map.insert(Slice("abc"), a));
    assert!(!map.insert(Owned("bcd".to_owned()), b));
    assert!(!map.insert(Slice("cde"), c));
    assert!(!map.insert(Owned("def".to_owned()), d));

    assert!(!map.insert(Owned("abc".to_owned()), a));
    assert!(!map.insert(Slice("bcd"), b));
    assert!(!map.insert(Owned("cde".to_owned()), c));
    assert!(!map.insert(Slice("def"), d));

    assert_eq!(map.find(&Slice("abc")), Some(&a));
    assert_eq!(map.find(&Slice("bcd")), Some(&b));
    assert_eq!(map.find(&Slice("cde")), Some(&c));
    assert_eq!(map.find(&Slice("def")), Some(&d));

    assert_eq!(map.find(&Owned("abc".to_owned())), Some(&a));
    assert_eq!(map.find(&Owned("bcd".to_owned())), Some(&b));
    assert_eq!(map.find(&Owned("cde".to_owned())), Some(&c));
    assert_eq!(map.find(&Owned("def".to_owned())), Some(&d));

    assert!(map.pop(&Slice("foo")).is_some());
    assert_eq!(map.move_iter().map(|(k, v)| format_strbuf!("{}{}", k, v))
                              .collect::<Vec<StrBuf>>()
                              .concat(),
               "abc50bcd51cde52def53".to_owned());
}
