// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// @has smoke/index.html

//! Very docs

// @has smoke/bar/index.html
pub mod bar {

    /// So correct
    // @has smoke/bar/baz/index.html
    pub mod baz {
        /// Much detail
        // @has smoke/bar/baz/fn.baz.html
        pub fn baz() { }
    }

    /// *wow*
    // @has smoke/bar/trait.Doge.html
    pub trait Doge { fn dummy(&self) { } }

    // @has smoke/bar/struct.Foo.html
    pub struct Foo { x: isize, y: usize }

    // @has smoke/bar/fn.prawns.html
    pub fn prawns((a, b): (isize, usize), Foo { x, y }: Foo) { }
}
