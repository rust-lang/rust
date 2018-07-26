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

// ignore-tidy-linelength

// @has foo/struct.Foo.html '//*[@class="docblock"]/p/a[@href="https://doc.rust-lang.org/nightly/std/primitive.u32.html"]' 'u32'
// @has foo/struct.Foo.html '//*[@class="docblock"]/p/a[@href="https://doc.rust-lang.org/nightly/std/primitive.i64.html"]' 'i64'

/// It contains [`u32`] and [i64].
pub struct Foo;
