// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::strbuf::StrBuf;

fn test_stack_assign() {
    let s: ~str = "a".to_owned();
    println!("{}", s.clone());
    let t: ~str = "a".to_owned();
    assert!(s == t);
    let u: ~str = "b".to_owned();
    assert!((s != u));
}

fn test_heap_lit() { "a big string".to_owned(); }

fn test_heap_assign() {
    let s: ~str = "a big ol' string".to_owned();
    let t: ~str = "a big ol' string".to_owned();
    assert!(s == t);
    let u: ~str = "a bad ol' string".to_owned();
    assert!((s != u));
}

fn test_heap_log() { let s = "a big ol' string".to_owned(); println!("{}", s); }

fn test_stack_add() {
    assert_eq!("a".to_owned() + "b", "ab".to_owned());
    let s: ~str = "a".to_owned();
    assert_eq!(s + s, "aa".to_owned());
    assert_eq!("".to_owned() + "", "".to_owned());
}

fn test_stack_heap_add() { assert!(("a".to_owned() + "bracadabra" == "abracadabra".to_owned())); }

fn test_heap_add() {
    assert_eq!("this should".to_owned() + " totally work", "this should totally work".to_owned());
}

fn test_append() {
    let mut s = StrBuf::new();
    s.push_str("a");
    assert_eq!(s.as_slice(), "a");

    let mut s = StrBuf::from_str("a");
    s.push_str("b");
    println!("{}", s.clone());
    assert_eq!(s.as_slice(), "ab");

    let mut s = StrBuf::from_str("c");
    s.push_str("offee");
    assert!(s.as_slice() == "coffee");

    s.push_str("&tea");
    assert!(s.as_slice() == "coffee&tea");
}

pub fn main() {
    test_stack_assign();
    test_heap_lit();
    test_heap_assign();
    test_heap_log();
    test_stack_add();
    test_stack_heap_add();
    test_heap_add();
    test_append();
}
