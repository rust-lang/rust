// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "foo"]

#[doc(hidden)]
pub mod hidden {
    pub struct Foo;

    impl Foo {
        #[doc(hidden)]
        pub fn this_should_be_hidden() {}
    }

    pub struct Bar;

    impl Bar {
        fn this_should_be_hidden() {}
    }
}

// @has foo/struct.Foo.html
// @!has - 'Methods'
// @!has - 'impl Foo'
// @!has - 'this_should_be_hidden'
pub use hidden::Foo;

// @has foo/struct.Bar.html
// @!has - 'Methods'
// @!has - 'impl Bar'
// @!has - 'this_should_be_hidden'
pub use hidden::Bar;
