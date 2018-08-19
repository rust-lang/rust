// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test checks for namespace pollution by private tests.
// Tests used to marked as public causing name conflicts with normal
// functions only in test builds.

// compile-flags: --test

mod a {
    pub fn foo() -> bool {
        true
    }
}

mod b {
    #[test]
    fn foo() {
        local_name(); // ensure the local name still works
    }

    #[test]
    fn local_name() {}
}

use a::*;
use b::*;

pub fn conflict() {
    let _: bool = foo();
}
