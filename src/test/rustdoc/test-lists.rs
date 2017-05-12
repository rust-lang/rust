// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "foo"]

// @has foo/fn.f.html
// @has - //ol/li "list"
// @has - //ol/li/ol/li "fooooo"
// @has - //ol/li/ol/li "x"
// @has - //ol/li "foo"
/// 1. list
///     1. fooooo
///     2. x
/// 2. foo
pub fn f() {}

// @has foo/fn.foo2.html
// @has - //ul/li "normal list"
// @has - //ul/li/ul/li "sub list"
// @has - //ul/li/ul/li "new elem still same elem and again same elem!"
// @has - //ul/li "new big elem"
/// * normal list
///     * sub list
///     * new elem
///       still same elem
///
///       and again same elem!
/// * new big elem
pub fn foo2() {}
