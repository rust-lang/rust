// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #48132. This was failing due to problems around
// the projection caching and dropck type enumeration.

// run-pass

#![feature(nll)]
#![allow(warnings)]

pub struct Container<T: Iterator> {
    value: Option<T::Item>,
}

impl<T: Iterator> Container<T> {
    pub fn new(iter: T) -> Self {
        panic!()
    }
}

pub struct Wrapper<'a> {
    content: &'a Content,
}

impl<'a, 'de> Wrapper<'a> {
    pub fn new(content: &'a Content) -> Self {
        Wrapper {
            content: content,
        }
    }
}

pub struct Content;

fn crash_it(content: Content) {
    let items = vec![content];
    let map = items.iter().map(|ref o| Wrapper::new(o));

    let mut map_visitor = Container::new(map);

}

fn main() {}
