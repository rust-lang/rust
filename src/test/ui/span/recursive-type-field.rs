// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::rc::Rc;

struct Foo<'a> { //~ ERROR recursive type
    bar: Bar<'a>,
    b: Rc<Bar<'a>>,
}

struct Bar<'a> { //~ ERROR recursive type
    y: (Foo<'a>, Foo<'a>),
    z: Option<Bar<'a>>,
    a: &'a Foo<'a>,
    c: &'a [Bar<'a>],
    d: [Bar<'a>; 1],
    e: Foo<'a>,
    x: Bar<'a>,
}

fn main() {}
