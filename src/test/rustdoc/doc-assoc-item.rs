// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct Foo<T> {
    x: T,
}

pub trait Bar {
    type Fuu;

    fn foo(foo: Self::Fuu);
}

// @has doc_assoc_item/struct.Foo.html '//*[@class="impl"]' 'impl<T: Bar<Fuu = u32>> Foo<T>'
impl<T: Bar<Fuu = u32>> Foo<T> {
    pub fn new(t: T) -> Foo<T> {
        Foo {
            x: t,
        }
    }
}
