// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "foo"]

// ignore-tidy-linelength

// @has foo/index.html '//*[@class="docblock"]/p/a[@href="../foo/struct.Foo.html#structfield.bar"]' 'Foo::bar'
// @has foo/index.html '//*[@class="docblock"]/p/a[@href="../foo/union.Bar.html#structfield.foo"]' 'Bar::foo'
// @has foo/index.html '//*[@class="docblock"]/p/a[@href="../foo/enum.Uniooon.html#X.v"]' 'Uniooon::X'

//! Test with [Foo::bar], [Bar::foo], [Uniooon::X]

pub struct Foo {
    pub bar: usize,
}

pub union Bar {
    pub foo: u32,
}

pub enum Uniooon {
    F,
    X,
    Y,
}
