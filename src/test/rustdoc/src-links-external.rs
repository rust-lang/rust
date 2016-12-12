// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:src-links-external.rs
// build-aux-docs
// ignore-cross-compile
// ignore-tidy-linelength

#![crate_name = "foo"]

extern crate src_links_external;

// @has foo/bar/index.html '//a/@href' '../../src/src_links_external/src-links-external.rs.html#11'
pub use src_links_external as bar;

// @has foo/bar/struct.Foo.html '//a/@href' '../../src/src_links_external/src-links-external.rs.html#11'
