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

pub trait Foo {
    // @has foo/trait.Foo.html '//h3[@id="tymethod.foo"]//div[@class="docblock attributes"]' '#[must_use]'
    #[must_use]
    fn foo();
}

#[must_use]
pub struct Bar;

impl Bar {
    // @has foo/struct.Bar.html '//h4[@id="method.bar"]//div[@class="docblock attributes"]' '#[must_use]'
    #[must_use]
    pub fn bar() {}

    // @has foo/struct.Bar.html '//h4[@id="method.bar2"]//div[@class="docblock attributes"]' '#[must_use]'
    #[must_use]
    pub fn bar2() {}
}
