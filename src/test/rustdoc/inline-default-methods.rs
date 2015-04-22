// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:inline-default-methods.rs
// ignore-cross-compile

extern crate inline_default_methods;

// @has inline_default_methods/trait.Foo.html
// @has - '//*[@class="rust trait"]' 'fn bar(&self);'
// @has - '//*[@class="rust trait"]' 'fn foo(&mut self) { ... }'
pub use inline_default_methods::Foo;
