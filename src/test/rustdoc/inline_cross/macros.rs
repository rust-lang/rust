// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:macros.rs
// build-aux-docs

#![feature(macro_test)]
#![feature(use_extern_macros)]

#![crate_name = "foo"]

extern crate macros;

// @has foo/index.html '//*[@class="docblock-short"]' '[Deprecated] [Experimental]'

// @has foo/macro.my_macro.html
// @has - '//*[@class="docblock"]' 'docs for my_macro'
// @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.2.3: text'
// @has - '//*[@class="stab unstable"]' 'macro_test'
// @has - '//a/@href' '../src/macros/macros.rs.html#19-21'
pub use macros::my_macro;
