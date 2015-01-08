// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]

struct StrWrap {
    s: String
}

impl StrWrap {
    fn new(s: &str) -> StrWrap {
        StrWrap { s: s.to_string() }
    }

    fn get_s<'a>(&'a self) -> &'a str {
        self.s.as_slice()
    }
}

struct MyStruct {
    s: StrWrap
}

impl MyStruct {
    fn new(s: &str) -> MyStruct {
        MyStruct { s: StrWrap::new(s) }
    }

    fn get_str_wrap<'a>(&'a self) -> &'a StrWrap {
        &self.s
    }
}

trait Matcher<T> {
    fn matches(&self, actual: T) -> bool;
}

fn assert_that<T, U: Matcher<T>>(actual: T, matcher: &U) {
    assert!(matcher.matches(actual));
}

struct EqualTo<T> {
    expected: T
}

impl<T: Eq> Matcher<T> for EqualTo<T> {
    fn matches(&self, actual: T) -> bool {
        self.expected.eq(&actual)
    }
}

fn equal_to<T: Eq>(expected: T) -> Box<EqualTo<T>> {
    box EqualTo { expected: expected }
}

pub fn main() {
    let my_struct = MyStruct::new("zomg");
    let s = my_struct.get_str_wrap();

    assert_that(s.get_s(), &*equal_to("zomg"));
}
