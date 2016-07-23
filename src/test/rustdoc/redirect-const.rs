// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name="foo"]

pub use hidden::STATIC_FOO;
pub use hidden::CONST_FOO;

mod hidden {
    // @has foo/hidden/STATIC_FOO.v.html
    // @has - '//p/a' '../../foo/STATIC_FOO.v.html'
    pub static STATIC_FOO: u64 = 0;
    // @has foo/hidden/CONST_FOO.v.html
    // @has - '//p/a' '../../foo/CONST_FOO.v.html'
    pub const CONST_FOO: u64 = 0;
}
