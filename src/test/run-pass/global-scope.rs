// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast

pub fn f() -> int { return 1; }

pub mod foo {
    pub fn f() -> int { return 2; }
    pub fn g() { assert!((f() == 2)); assert!((::f() == 1)); }
}

pub fn main() { return foo::g(); }
