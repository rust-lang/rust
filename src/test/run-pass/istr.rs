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
    let s: StrBuf = "a".to_strbuf();
    println!("{}", s.clone());
    let t: StrBuf = "a".to_strbuf();
    assert!(s == t);
    let u: StrBuf = "b".to_strbuf();
    assert!((s != u));
}

fn test_heap_lit() { "a big string".to_strbuf(); }

fn test_heap_assign() {
    let s: StrBuf = "a big ol' string".to_strbuf();
    let t: StrBuf = "a big ol' string".to_strbuf();
    assert!(s == t);
    let u: StrBuf = "a bad ol' string".to_strbuf();
    assert!((s != u));
}

fn test_heap_log() {
    let s = "a big ol' string".to_strbuf();
    println!("{}", s);
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
    test_append();
}
