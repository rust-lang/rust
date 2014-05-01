// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unnecessary_allocation)]

fn f1(ref_string: &str) -> ~str {
    match ref_string {
        "a" => "found a".to_owned(),
        "b" => "found b".to_owned(),
        _ => "not found".to_owned()
    }
}

fn f2(ref_string: &str) -> ~str {
    match ref_string {
        "a" => "found a".to_owned(),
        "b" => "found b".to_owned(),
        s => format!("not found ({})", s)
    }
}

fn g1(ref_1: &str, ref_2: &str) -> ~str {
    match (ref_1, ref_2) {
        ("a", "b") => "found a,b".to_owned(),
        ("b", "c") => "found b,c".to_owned(),
        _ => "not found".to_owned()
    }
}

fn g2(ref_1: &str, ref_2: &str) -> ~str {
    match (ref_1, ref_2) {
        ("a", "b") => "found a,b".to_owned(),
        ("b", "c") => "found b,c".to_owned(),
        (s1, s2) => format!("not found ({}, {})", s1, s2)
    }
}

pub fn main() {
    assert_eq!(f1("b".to_owned()), "found b".to_owned());
    assert_eq!(f1("c"), "not found".to_owned());
    assert_eq!(f1("d"), "not found".to_owned());
    assert_eq!(f2("b".to_owned()), "found b".to_owned());
    assert_eq!(f2("c"), "not found (c)".to_owned());
    assert_eq!(f2("d"), "not found (d)".to_owned());
    assert_eq!(g1("b".to_owned(), "c".to_owned()), "found b,c".to_owned());
    assert_eq!(g1("c", "d"), "not found".to_owned());
    assert_eq!(g1("d", "e"), "not found".to_owned());
    assert_eq!(g2("b".to_owned(), "c".to_owned()), "found b,c".to_owned());
    assert_eq!(g2("c", "d"), "not found (c, d)".to_owned());
    assert_eq!(g2("d", "e"), "not found (d, e)".to_owned());
}
