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

mod hidden {
    // @has foo/hidden/struct.Foo.html
    // @has - '//p/a' '../../foo/struct.FooBar.html'
    pub struct Foo {}

    // @has foo/hidden/bar/index.html
    // @has - '//p/a' '../../foo/baz/index.html'
    pub mod bar {
        // @has foo/hidden/bar/struct.Thing.html
        // @has - '//p/a' '../../foo/baz/struct.Thing.html'
        pub struct Thing {}
    }
}

// @has foo/struct.FooBar.html
pub use hidden::Foo as FooBar;

// @has foo/baz/index.html
// @has foo/baz/struct.Thing.html
pub use hidden::bar as baz;
