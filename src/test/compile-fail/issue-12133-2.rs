// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-12133-rlib.rs
// aux-build:issue-12133-dylib.rs
// no-prefer-dynamic

// error-pattern: dependencies were not all found in either dylib or rlib format

extern crate a = "issue-12133-rlib";
extern crate b = "issue-12133-dylib";

fn main() {}
