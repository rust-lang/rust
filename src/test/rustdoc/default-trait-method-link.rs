// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "foo"]

// @has foo/trait.Foo.html '//a[@href="../foo/trait.Foo.html#tymethod.req"]' 'req'
// @has foo/trait.Foo.html '//a[@href="../foo/trait.Foo.html#method.prov"]' 'prov'

/// Always make sure to implement [`req`], but you don't have to implement [`prov`].
///
/// [`req`]: Foo::req
/// [`prov`]: Foo::prov
pub trait Foo {
    /// Required
    fn req();
    /// Provided
    fn prov() {}
}
