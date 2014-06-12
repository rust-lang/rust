// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]

use std::gc::{Gc, GC};

trait Foo {
    fn foo(&self) -> String;
}

impl<T:Foo> Foo for Gc<T> {
    fn foo(&self) -> String {
        format!("box(GC) {}", (**self).foo())
    }
}

impl Foo for uint {
    fn foo(&self) -> String {
        format!("{}", *self)
    }
}

pub fn main() {
    let x = box(GC) 3u;
    assert_eq!(x.foo(), "box(GC) 3".to_string());
}
