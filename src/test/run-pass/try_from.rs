// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test relies on `TryFrom` being auto impl for all `T: Into`
// and `TryInto` being auto impl for all `U: TryFrom`

// This test was added to show the motivation for doing this
// over `TryFrom` being auto impl for all `T: From`

#![feature(try_from, never_type)]

use std::convert::TryInto;

struct Foo<T> {
    t: T
}

/*
// This fails to compile due to coherence restrictions
// as of rust version 1.32.x
impl<T> From<Foo<T>> for Box<T> {
    fn from(foo: Foo<T>) -> Box<T> {
        Box::new(foo.t)
    }
}
*/

impl<T> Into<Box<T>> for Foo<T> {
    fn into(self) -> Box<T> {
        Box::new(self.t)
    }
}

pub fn main() {
    let _: Result<Box<i32>, !> = Foo { t: 10 }.try_into();
}
