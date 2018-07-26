// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "lib"]

extern crate my_crate;

pub fn g() {} // (a)

#[macro_export]
macro_rules! unhygienic_macro {
    () => {
        // (1) unhygienic: depends on `my_crate` in the crate root at the invocation site.
        ::my_crate::f();

        // (2) unhygienic: defines `f` at the invocation site (in addition to the above point).
        use my_crate::f;
        f();

        g(); // (3) unhygienic: `g` needs to be in scope at use site.

        $crate::g(); // (4) hygienic: this always resolves to (a)
    }
}

#[allow(unused)]
fn test_unhygienic() {
    unhygienic_macro!();
    f(); // `f` was defined at the use site
}
