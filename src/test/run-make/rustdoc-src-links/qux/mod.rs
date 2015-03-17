// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Dox
// @has src/foo/qux/mod.rs.html
// @has foo/qux/index.html '//a/@href' '../../src/foo/qux/mod.rs.html'

// @has foo/qux/bar/index.html '//a/@href' '../../../src/foo/qux/mod.rs.html'
pub mod bar {

    /// Dox
    // @has foo/qux/bar/baz/index.html '//a/@href' '../../../../src/foo/qux/mod.rs.html'
    pub mod baz {
        /// Dox
        // @has foo/qux/bar/baz/fn.baz.html '//a/@href' '../../../../src/foo/qux/mod.rs.html'
        pub fn baz() { }
    }

    /// Dox
    // @has foo/qux/bar/trait.Foobar.html '//a/@href' '../../../src/foo/qux/mod.rs.html'
    pub trait Foobar { fn dummy(&self) { } }

    // @has foo/qux/bar/struct.Foo.html '//a/@href' '../../../src/foo/qux/mod.rs.html'
    pub struct Foo { x: i32, y: u32 }

    // @has foo/qux/bar/fn.prawns.html '//a/@href' '../../../src/foo/qux/mod.rs.html'
    pub fn prawns((a, b): (i32, u32), Foo { x, y }: Foo) { }
}

/// Dox
// @has foo/qux/fn.modfn.html '//a/@href' '../../src/foo/qux/mod.rs.html'
pub fn modfn() { }
