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

enum Token {
    Text(@StrBuf),
    ETag(@Vec<StrBuf> , @StrBuf),
    UTag(@Vec<StrBuf> , @StrBuf),
    Section(@Vec<StrBuf> , bool, @Vec<Token> , @StrBuf, @StrBuf, @StrBuf, @StrBuf, @StrBuf),
    IncompleteSection(@Vec<StrBuf> , bool, @StrBuf, bool),
    Partial(@StrBuf, @StrBuf, @StrBuf),
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
// assert!(check_strs(fmt!("%?", Text(@"foo".to_owned())), "Text(@~\"foo\")"));
// assert!(check_strs(fmt!("%?", ETag(@~["foo".to_owned()], @"bar".to_owned())),
//                    "ETag(@~[ ~\"foo\" ], @~\"bar\")"));

    let t = Text(@"foo".to_strbuf());
    let u = Section(@vec!("alpha".to_strbuf()),
                          true,
                          @vec!(t),
                          @"foo".to_strbuf(),
                    @"foo".to_strbuf(), @"foo".to_strbuf(), @"foo".to_strbuf(),
                    @"foo".to_strbuf());
    let v = format!("{:?}", u);    // this is the line that causes the seg fault
    assert!(v.len() > 0);
}
