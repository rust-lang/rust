// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// @has foo/index.html
#![crate_name = "foo"]

//! Very docs

// @has foo/bar/index.html
pub mod bar {

    /// So correct
    // @has foo/bar/baz/index.html
    pub mod baz {
        /// Much detail
        // @has foo/bar/baz/fn.baz.html
        pub fn baz() { }
    }

    /// *wow*
    // @has foo/bar/trait.Doge.html
    pub trait Doge { fn dummy(&self) { } }

    // @has foo/bar/struct.Foo.html
    pub struct Foo { x: int, y: uint }

    // @has foo/bar/fn.prawns.html
    pub fn prawns((a, b): (int, uint), Foo { x, y }: Foo) { }
}
