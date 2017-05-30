// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![forbid(dead_code)]

// The most simple case: support one layer
// Issue #42303 was originally reported about this issue
mod a {
    pub struct Foo;

    type Bar = Foo;

    impl Bar {
        pub fn baz(&self) {}
    }
}

// Multiple layers of re-defining,
// to make sure we don't regress here
mod b {
    pub struct Foo;

    type Bar = Foo;
    type FooBar = Bar;
    type Baz = FooBar;
    type FooBarBaz = Baz;

    impl FooBarBaz {
        pub fn baz(&self) {}
    }
}

fn main() {
    a::Foo.baz();
    b::Foo.baz();
}
