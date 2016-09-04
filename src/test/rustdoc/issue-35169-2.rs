// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Deref;
use std::ops::DerefMut;

pub struct Foo;
pub struct Bar;

impl Foo {
    pub fn by_ref(&self) {}
    pub fn by_explicit_ref(self: &Foo) {}
    pub fn by_mut_ref(&mut self) {}
    pub fn by_explicit_mut_ref(self: &mut Foo) {}
    pub fn static_foo() {}
}

impl Deref for Bar {
    type Target = Foo;
    fn deref(&self) -> &Foo { loop {} }
}

impl DerefMut for Bar {
    fn deref_mut(&mut self) -> &mut Foo { loop {} }
}

// @has issue_35169_2/Bar.t.html
// @has issue_35169_2/struct.Bar.html
// @has - '//*[@id="by_ref.v"]' 'fn by_ref(&self)'
// @has - '//*[@id="method.by_ref"]' 'fn by_ref(&self)'
// @has - '//*[@id="by_explicit_ref.v"]' 'fn by_explicit_ref(self: &Foo)'
// @has - '//*[@id="method.by_explicit_ref"]' 'fn by_explicit_ref(self: &Foo)'
// @has - '//*[@id="by_mut_ref.v"]' 'fn by_mut_ref(&mut self)'
// @has - '//*[@id="method.by_mut_ref"]' 'fn by_mut_ref(&mut self)'
// @has - '//*[@id="by_explicit_mut_ref.v"]' 'fn by_explicit_mut_ref(self: &mut Foo)'
// @has - '//*[@id="method.by_explicit_mut_ref"]' 'fn by_explicit_mut_ref(self: &mut Foo)'
// @!has - '//*[@id="static_foo.v"]' 'fn static_foo()'
// @!has - '//*[@id="method.static_foo"]' 'fn static_foo()'
