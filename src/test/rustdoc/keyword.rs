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

#![feature(doc_keyword)]

// @has foo/index.html '//h2[@id="keywords"]' 'Keywords'
// @has foo/index.html '//a[@href="keyword.match.html"]' 'match'
// @has foo/keyword.match.html '//a[@class="keyword"]' 'match'
// @has foo/keyword.match.html '//section[@id="main"]//div[@class="docblock"]//p' 'this is a test!'
// @!has foo/index.html '//a/@href' 'foo/index.html'
// @!has foo/foo/index.html
// @!has-dir foo/foo
#[doc(keyword = "match")]
/// this is a test!
mod foo{}
