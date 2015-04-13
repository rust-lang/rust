// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test is ensuring that the `ensure_not_fmt_string_literal!`
// macro cannot be invoked (at least not in code whose expansion ends
// up in the expanded output) without the appropriate feature.

pub fn f0() {
    panic!("this should work");
}

pub fn main() {
    __unstable_rustc_ensure_not_fmt_string_literal!(
        "`main`", "this should work, but its unstable");
    //~^^ ERROR use of unstable library feature 'ensure_not_fmt_string_literal'
    //~| HELP add #![feature(ensure_not_fmt_string_literal)] to the crate attributes to enable
    f0();
}
