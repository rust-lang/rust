// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


extern crate debug;

use std::gc::{Gc, GC};

enum Token {
    Text(Gc<String>),
    ETag(Gc<Vec<String>> , Gc<String>),
    UTag(Gc<Vec<String>> , Gc<String>),
    Section(Gc<Vec<String>> , bool, Gc<Vec<Token>>, Gc<String>,
            Gc<String>, Gc<String>, Gc<String>, Gc<String>),
    IncompleteSection(Gc<Vec<String>> , bool, Gc<String>, bool),
    Partial(Gc<String>, Gc<String>, Gc<String>),
}

fn check_strs(actual: &str, expected: &str) -> bool
{
    if actual != expected
    {
        println!("Found {}, but expected {}", actual, expected);
        return false;
    }
    return true;
}

pub fn main()
{
// assert!(check_strs(fmt!("%?", Text(@"foo".to_string())), "Text(@~\"foo\")"));
// assert!(check_strs(fmt!("%?", ETag(@~["foo".to_string()], @"bar".to_string())),
//                    "ETag(@~[ ~\"foo\" ], @~\"bar\")"));

    let t = Text(box(GC) "foo".to_string());
    let u = Section(box(GC) vec!("alpha".to_string()),
                          true,
                          box(GC) vec!(t),
                          box(GC) "foo".to_string(),
                    box(GC) "foo".to_string(), box(GC) "foo".to_string(), box(GC) "foo".to_string(),
                    box(GC) "foo".to_string());
    let v = format!("{:?}", u);    // this is the line that causes the seg fault
    assert!(v.len() > 0);
}
