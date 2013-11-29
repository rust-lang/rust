// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo { foo: bool, bar: Option<int>, baz: int }

pub fn main() {
    match @Foo{foo: true, bar: Some(10), baz: 20} {
      @Foo{foo: true, bar: Some(_), ..} => {}
      @Foo{foo: false, bar: None, ..} => {}
      @Foo{foo: true, bar: None, ..} => {}
      @Foo{foo: false, bar: Some(_), ..} => {}
    }
}
