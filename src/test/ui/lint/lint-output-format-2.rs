// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:lint_output_format.rs

#![feature(unstable_test_feature)]
#![feature(rustc_attrs)]

extern crate lint_output_format;
use lint_output_format::{foo, bar};
//~^ WARNING use of deprecated item 'lint_output_format::foo': text

#[rustc_error]
fn main() { //~ ERROR: compilation successful
    let _x = foo();
    //~^ WARNING use of deprecated item 'lint_output_format::foo': text
    let _y = bar();
}
