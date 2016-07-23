// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// @has issue_19055/Any.t.html
pub trait Any {}

impl<'any> Any + 'any {
    // @has - '//*[@id="is.v"]' 'fn is'
    pub fn is<T: 'static>(&self) -> bool { loop {} }

    // @has - '//*[@id="downcast_ref.v"]' 'fn downcast_ref'
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> { loop {} }

    // @has - '//*[@id="downcast_mut.v"]' 'fn downcast_mut'
    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> { loop {} }
}

pub trait Foo {
    fn foo(&self) {}
}

// @has - '//*[@id="foo.v"]' 'fn foo'
impl Foo for Any {}
