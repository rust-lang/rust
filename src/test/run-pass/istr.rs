// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn test_stack_assign() {
    let s: ~str = ~"a";
    info2!("{}", s.clone());
    let t: ~str = ~"a";
    assert!(s == t);
    let u: ~str = ~"b";
    assert!((s != u));
}

fn test_heap_lit() { ~"a big string"; }

fn test_heap_assign() {
    let s: ~str = ~"a big ol' string";
    let t: ~str = ~"a big ol' string";
    assert!(s == t);
    let u: ~str = ~"a bad ol' string";
    assert!((s != u));
}

fn test_heap_log() { let s = ~"a big ol' string"; info2!("{}", s); }

fn test_stack_add() {
    assert_eq!(~"a" + "b", ~"ab");
    let s: ~str = ~"a";
    assert_eq!(s + s, ~"aa");
    assert_eq!(~"" + "", ~"");
}

fn test_stack_heap_add() { assert!((~"a" + "bracadabra" == ~"abracadabra")); }

fn test_heap_add() {
    assert_eq!(~"this should" + " totally work", ~"this should totally work");
}

fn test_append() {
    let mut s = ~"";
    s.push_str("a");
    assert_eq!(s, ~"a");

    let mut s = ~"a";
    s.push_str("b");
    info2!("{}", s.clone());
    assert_eq!(s, ~"ab");

    let mut s = ~"c";
    s.push_str("offee");
    assert!(s == ~"coffee");

    s.push_str("&tea");
    assert!(s == ~"coffee&tea");
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
