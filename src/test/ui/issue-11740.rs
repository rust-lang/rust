// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]
#![allow(warnings)]

struct Attr {
    name: String,
    value: String,
}

struct Element {
    attrs: Vec<Box<Attr>>,
}

impl Element {
    pub unsafe fn get_attr<'a>(&'a self, name: &str) {
        self.attrs
            .iter()
            .find(|attr| {
                      let attr: &&Box<Attr> = std::mem::transmute(attr);
                      true
                  });
    }
}

#[rustc_error]
fn main() { //~ ERROR compilation successful
    let element = Element { attrs: Vec::new() };
    let _ = unsafe { element.get_attr("foo") };
}
