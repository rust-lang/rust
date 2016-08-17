// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Deref;

pub struct Foo;
pub struct Bar;

impl Foo {
    pub fn foo(&self) {}
    pub fn static_foo() {}
}

impl Deref for Bar {
    type Target = Foo;
    fn deref(&self) -> &Foo { loop {} }
}

// @has issue_19190/Bar.t.html
// @has issue_19190/struct.Bar.html
// @has - '//*[@id="foo.v"]' 'fn foo(&self)'
// @has - '//*[@id="method.foo"]' 'fn foo(&self)'
// @!has - '//*[@id="static_foo.v"]' 'fn static_foo()'
// @!has - '//*[@id="method.static_foo"]' 'fn static_foo()'
