// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// compile-flags: --document-private-items

#![crate_name = "foo"]

// @has foo/fn.bar.html
// @has - '//*[@class="rust fn"]' 'pub fn bar() -> '
/// foo
pub fn bar() -> usize {
    2
}

// @has foo/struct.Foo.html
// @has - '//*[@class="method"]' 'pub fn new()'
// @has - '//*[@class="method"]' 'fn not_pub()'
pub struct Foo(usize);

impl Foo {
    pub fn new() -> Foo { Foo(0) }
    fn not_pub() {}
}
