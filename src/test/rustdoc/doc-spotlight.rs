// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(doc_spotlight)]

pub struct Wrapper<T> {
    inner: T,
}

impl<T: SomeTrait> SomeTrait for Wrapper<T> {}

#[doc(spotlight)]
pub trait SomeTrait {
    // @has doc_spotlight/trait.SomeTrait.html
    // @has - '//code[@class="content"]' 'impl<T: SomeTrait> SomeTrait for Wrapper<T>'
    fn wrap_me(self) -> Wrapper<Self> where Self: Sized {
        Wrapper {
            inner: self,
        }
    }
}

pub struct SomeStruct;
impl SomeTrait for SomeStruct {}

impl SomeStruct {
    // @has doc_spotlight/struct.SomeStruct.html
    // @has - '//code[@class="content"]' 'impl SomeTrait for SomeStruct'
    // @has - '//code[@class="content"]' 'impl<T: SomeTrait> SomeTrait for Wrapper<T>'
    pub fn new() -> SomeStruct {
        SomeStruct
    }
}

// @has doc_spotlight/fn.bare_fn.html
// @has - '//code[@class="content"]' 'impl SomeTrait for SomeStruct'
pub fn bare_fn() -> SomeStruct {
    SomeStruct
}
