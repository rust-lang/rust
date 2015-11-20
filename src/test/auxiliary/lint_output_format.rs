// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name="lint_output_format"]
#![crate_type = "lib"]
#![feature(staged_api)]
#![staged_api]
#![unstable(feature = "test_feature", issue = "0")]

#[stable(feature = "test_feature", since = "1.0.0")]
#[rustc_deprecated(since = "1.0.0", reason = "text")]
pub fn foo() -> usize {
    20
}

#[unstable(feature = "test_feature", issue = "0")]
pub fn bar() -> usize {
    40
}

#[unstable(feature = "test_feature", issue = "0")]
pub fn baz() -> usize {
    30
}
