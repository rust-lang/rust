// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_type_defaults)]
#![feature(associated_consts)]

// @has issue_28478/Bar.t.html
pub trait Bar {
    // @has - '//*[@id="Bar.t"]' 'type Bar = ()'
    // @has - '//*[@href="#Bar.t"]' 'Bar'
    type Bar = ();
    // @has - '//*[@id="Baz.v"]' 'const Baz: usize = 7'
    // @has - '//*[@href="#Baz.v"]' 'Baz'
    const Baz: usize = 7;
    // @has - '//*[@id="bar.v"]' 'fn bar'
    fn bar();
    // @has - '//*[@id="baz.v"]' 'fn baz'
    fn baz() { }
}

// @has issue_28478/Foo.t.html
pub struct Foo;

impl Foo {
    // @has - '//*[@href="#foo.v"]' 'foo'
    pub fn foo() {}
}

impl Bar for Foo {
    // @has - '//*[@href="../issue_28478/Bar.t.html#Bar.t"]' 'Bar'
    // @has - '//*[@href="../issue_28478/Bar.t.html#Baz.v"]' 'Baz'
    // @has - '//*[@href="../issue_28478/Bar.t.html#bar.v"]' 'bar'
    fn bar() {}
    // @has - '//*[@href="../issue_28478/Bar.t.html#baz.v"]' 'baz'
}
