// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "rustdoc_test"]

// @has search-index.js Foo
pub use private::Foo;

mod private {
    pub struct Foo;
    impl Foo {
        pub fn test_method() {} // @has - test_method
        fn priv_method() {} // @!has - priv_method
    }

    pub trait PrivateTrait {
        fn trait_method(&self) {} // @!has - priv_method
    }
}
