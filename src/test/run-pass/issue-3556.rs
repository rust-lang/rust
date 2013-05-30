// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;
use std::io::WriterUtil;

enum Token {
        Text(@~str),
        ETag(@~[~str], @~str),
        UTag(@~[~str], @~str),
        Section(@~[~str], bool, @~[Token], @~str, @~str, @~str, @~str, @~str),
        IncompleteSection(@~[~str], bool, @~str, bool),
        Partial(@~str, @~str, @~str),
}

fn check_strs(actual: &str, expected: &str) -> bool
{
        if actual != expected
        {
            io::stderr().write_line(fmt!("Found %s, but expected %s", actual, expected));
            return false;
        }
        return true;
}

pub fn main()
{
 //       assert!(check_strs(fmt!("%?", Text(@~"foo")), "Text(@~\"foo\")"));
 //       assert!(check_strs(fmt!("%?", ETag(@~[~"foo"], @~"bar")), "ETag(@~[ ~\"foo\" ], @~\"bar\")"));

        let t = Text(@~"foo");
        let u = Section(@~[~"alpha"], true, @~[t], @~"foo", @~"foo", @~"foo", @~"foo", @~"foo");
        let v = fmt!("%?", u);    // this is the line that causes the seg fault
        assert!(v.len() > 0);
}
