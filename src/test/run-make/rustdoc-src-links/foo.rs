// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "foo"]

//! Dox
// @has src/foo/foo.rs.html
// @has foo/index.html '//a/@href' '../src/foo/foo.rs.html'

pub mod qux;

// @has foo/bar/index.html '//a/@href' '../../src/foo/foo.rs.html'
pub mod bar {

    /// Dox
    // @has foo/bar/baz/index.html '//a/@href' '../../../src/foo/foo.rs.html'
    pub mod baz {
        /// Dox
        // @has foo/bar/baz/fn.baz.html '//a/@href' '../../../src/foo/foo.rs.html'
        pub fn baz() { }
    }

    /// Dox
    // @has foo/bar/trait.Foobar.html '//a/@href' '../../src/foo/foo.rs.html'
    pub trait Foobar { fn dummy(&self) { } }

    // @has foo/bar/struct.Foo.html '//a/@href' '../../src/foo/foo.rs.html'
    pub struct Foo { x: i32, y: u32 }

    // @has foo/bar/fn.prawns.html '//a/@href' '../../src/foo/foo.rs.html'
    pub fn prawns((a, b): (i32, u32), Foo { x, y }: Foo) { }
}

/// Dox
// @has foo/fn.modfn.html '//a/@href' '../src/foo/foo.rs.html'
pub fn modfn() { }
