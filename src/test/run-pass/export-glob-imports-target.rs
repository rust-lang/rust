// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that a glob-export functions as an import
// when referenced within its own local scope.

// Modified to not use export since it's going away. --pcw

#[feature(globs)];

mod foo {
    use foo::bar::*;
    pub mod bar {
        pub static a : int = 10;
    }
    pub fn zum() {
        let _b = a;
    }
}

pub fn main() { }
