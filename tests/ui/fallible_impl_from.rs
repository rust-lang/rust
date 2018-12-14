// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(clippy::fallible_impl_from)]

// docs example
struct Foo(i32);
impl From<String> for Foo {
    fn from(s: String) -> Self {
        Foo(s.parse().unwrap())
    }
}

struct Valid(Vec<u8>);

impl<'a> From<&'a str> for Valid {
    fn from(s: &'a str) -> Valid {
        Valid(s.to_owned().into_bytes())
    }
}
impl From<usize> for Valid {
    fn from(i: usize) -> Valid {
        Valid(Vec::with_capacity(i))
    }
}

struct Invalid;

impl From<usize> for Invalid {
    fn from(i: usize) -> Invalid {
        if i != 42 {
            panic!();
        }
        Invalid
    }
}

impl From<Option<String>> for Invalid {
    fn from(s: Option<String>) -> Invalid {
        let s = s.unwrap();
        if !s.is_empty() {
            panic!(42);
        } else if s.parse::<u32>().unwrap() != 42 {
            panic!("{:?}", s);
        }
        Invalid
    }
}

trait ProjStrTrait {
    type ProjString;
}
impl<T> ProjStrTrait for Box<T> {
    type ProjString = String;
}
impl<'a> From<&'a mut <Box<u32> as ProjStrTrait>::ProjString> for Invalid {
    fn from(s: &'a mut <Box<u32> as ProjStrTrait>::ProjString) -> Invalid {
        if s.parse::<u32>().ok().unwrap() != 42 {
            panic!("{:?}", s);
        }
        Invalid
    }
}

struct Unreachable;

impl From<String> for Unreachable {
    fn from(s: String) -> Unreachable {
        if s.is_empty() {
            return Unreachable;
        }
        match s.chars().next() {
            Some(_) => Unreachable,
            None => unreachable!(), // do not lint the unreachable macro
        }
    }
}

fn main() {}
