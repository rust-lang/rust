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

// @has issue_28478/trait.Bar.html
pub trait Bar {
    // @has - '//*[@id="associatedtype.Bar"]' 'type Bar = ()'
    // @has - '//*[@href="#associatedtype.Bar"]' 'Bar'
    type Bar = ();
    // @has - '//*[@id="associatedconstant.Baz"]' 'const Baz: usize'
    // @has - '//*[@class="docblock"]' 'Baz: usize = 7'
    // @has - '//*[@href="#associatedconstant.Baz"]' 'Baz'
    const Baz: usize = 7;
    // @has - '//*[@id="tymethod.bar"]' 'fn bar'
    fn bar();
    // @has - '//*[@id="method.baz"]' 'fn baz'
    fn baz() { }
}

// @has issue_28478/struct.Foo.html
pub struct Foo;

impl Foo {
    // @has - '//*[@href="#method.foo"]' 'foo'
    pub fn foo() {}
}

impl Bar for Foo {
    // @has - '//*[@href="../issue_28478/trait.Bar.html#associatedtype.Bar"]' 'Bar'
    // @has - '//*[@href="../issue_28478/trait.Bar.html#associatedconstant.Baz"]' 'Baz'
    // @has - '//*[@href="../issue_28478/trait.Bar.html#tymethod.bar"]' 'bar'
    fn bar() {}
    // @has - '//*[@href="../issue_28478/trait.Bar.html#method.baz"]' 'baz'
}
