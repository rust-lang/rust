// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:empty.rs
// aux-build:variant-struct.rs
// ignore-cross-compile

// @has issue_33178_1/index.html
// @!has - //a/@title empty
pub extern crate empty;

// @!has - //a/@title variant_struct
pub extern crate variant_struct as foo;
