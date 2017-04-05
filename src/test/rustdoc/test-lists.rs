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

// ignore-tidy-linelength

// @has foo/fn.f.html
// @has - "<pre class='rust fn'>pub fn f()</pre><div class='docblock'><ol><li>list<ol><li>fooooo</li><li>x</li></ol></li><li>foo</li></ol>"
/// 1. list
///     1. fooooo
///     2. x
/// 2. foo
pub fn f() {}

// @has foo/fn.foo2.html
// @has - "<pre class='rust fn'>pub fn foo2()</pre><div class='docblock'><ul><li>normal list<ul><li><p>sub list</p></li><li><p>new elem still same elem</p><p>and again same elem!</p></li></ul></li><li>new big elem</li></ul>"
/// * normal list
///     * sub list
///     * new elem
///       still same elem
///
///       and again same elem!
/// * new big elem
pub fn foo2() {}