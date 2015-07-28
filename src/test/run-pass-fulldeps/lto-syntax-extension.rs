// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:lto-syntax-extension-lib.rs
// aux-build:lto-syntax-extension-plugin.rs
// compile-flags:-C lto
// ignore-stage1
// no-prefer-dynamic

#![feature(plugin)]
#![plugin(lto_syntax_extension_plugin)]

extern crate lto_syntax_extension_lib;

fn main() {
    lto_syntax_extension_lib::foo();
}
