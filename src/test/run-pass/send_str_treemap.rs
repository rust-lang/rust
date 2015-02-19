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

use self::collections::BTreeMap;
use std::borrow::{Cow, IntoCow};

type SendStr = Cow<'static, str>;

pub fn main() {
    let mut map: BTreeMap<SendStr, uint> = BTreeMap::new();
    assert!(map.insert("foo".into_cow(), 42).is_none());
    assert!(map.insert("foo".to_string().into_cow(), 42).is_some());
    assert!(map.insert("foo".into_cow(), 42).is_some());
    assert!(map.insert("foo".to_string().into_cow(), 42).is_some());

    assert!(map.insert("foo".into_cow(), 43).is_some());
    assert!(map.insert("foo".to_string().into_cow(), 44).is_some());
    assert!(map.insert("foo".into_cow(), 45).is_some());
    assert!(map.insert("foo".to_string().into_cow(), 46).is_some());

    let v = 46;

    assert_eq!(map.get(&"foo".to_string().into_cow()), Some(&v));
    assert_eq!(map.get(&"foo".into_cow()), Some(&v));

    let (a, b, c, d) = (50, 51, 52, 53);

    assert!(map.insert("abc".into_cow(), a).is_none());
    assert!(map.insert("bcd".to_string().into_cow(), b).is_none());
    assert!(map.insert("cde".into_cow(), c).is_none());
    assert!(map.insert("def".to_string().into_cow(), d).is_none());

    assert!(map.insert("abc".into_cow(), a).is_some());
    assert!(map.insert("bcd".to_string().into_cow(), b).is_some());
    assert!(map.insert("cde".into_cow(), c).is_some());
    assert!(map.insert("def".to_string().into_cow(), d).is_some());

    assert!(map.insert("abc".to_string().into_cow(), a).is_some());
    assert!(map.insert("bcd".into_cow(), b).is_some());
    assert!(map.insert("cde".to_string().into_cow(), c).is_some());
    assert!(map.insert("def".into_cow(), d).is_some());

    assert_eq!(map.get(&"abc".into_cow()), Some(&a));
    assert_eq!(map.get(&"bcd".into_cow()), Some(&b));
    assert_eq!(map.get(&"cde".into_cow()), Some(&c));
    assert_eq!(map.get(&"def".into_cow()), Some(&d));

    assert_eq!(map.get(&"abc".to_string().into_cow()), Some(&a));
    assert_eq!(map.get(&"bcd".to_string().into_cow()), Some(&b));
    assert_eq!(map.get(&"cde".to_string().into_cow()), Some(&c));
    assert_eq!(map.get(&"def".to_string().into_cow()), Some(&d));

    assert!(map.remove(&"foo".into_cow()).is_some());
    assert_eq!(map.into_iter().map(|(k, v)| format!("{}{}", k, v))
                              .collect::<Vec<String>>()
                              .concat(),
               "abc50bcd51cde52def53".to_string());
}
