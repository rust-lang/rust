// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:extern-links.rs
// ignore-cross-compile

#![crate_name = "foo"]

extern crate extern_links;

// @!has foo/index.html '//a' 'extern_links'
#[doc(no_inline)]
pub use extern_links as extern_links2;

// @!has foo/index.html '//a' 'Foo'
#[doc(no_inline)]
pub use extern_links::Foo;

#[doc(hidden)]
pub mod hidden {
    // @!has foo/hidden/extern_links/index.html
    // @!has foo/hidden/extern_links/struct.Foo.html
    pub use extern_links;
}
