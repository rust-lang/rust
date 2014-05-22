// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]

extern crate debug;

enum Token {
    Text(@String),
    ETag(@Vec<String> , @String),
    UTag(@Vec<String> , @String),
    Section(@Vec<String> , bool, @Vec<Token> , @String, @String, @String, @String, @String),
    IncompleteSection(@Vec<String> , bool, @String, bool),
    Partial(@String, @String, @String),
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

    let t = Text(@"foo".to_string());
    let u = Section(@vec!("alpha".to_string()),
                          true,
                          @vec!(t),
                          @"foo".to_string(),
                    @"foo".to_string(), @"foo".to_string(), @"foo".to_string(),
                    @"foo".to_string());
    let v = format!("{:?}", u);    // this is the line that causes the seg fault
    assert!(v.len() > 0);
}
