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
pub mod bar {

    /// Dox
    pub mod baz {
        /// Dox
        pub fn baz() { }
    }

    /// Dox
    pub trait Foobar { fn dummy(&self) { } }

    pub struct Foo { x: i32, y: u32 }

    pub fn prawns((a, b): (i32, u32), Foo { x, y }: Foo) { }
}

/// Dox
pub fn modfn() { }
