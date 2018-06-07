// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(staged_api)]

#![stable(feature = "rust1", since = "1.0.0")]

/// docs for my_macro
#[unstable(feature = "macro_test", issue = "0")]
#[rustc_deprecated(since = "1.2.3", reason = "text")]
#[macro_export]
macro_rules! my_macro {
    () => ()
}
