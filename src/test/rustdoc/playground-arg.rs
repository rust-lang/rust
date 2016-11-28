// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --playground-url=https://example.com/ -Z unstable-options
// ignore-tidy-linelength

#![crate_name = "foo"]

//! ```
//! use foo::dummy;
//! dummy();
//! ```

pub fn dummy() {}

// ensure that `extern crate foo;` was inserted into code snips automatically:
// @matches foo/index.html '//a[@class="test-arrow"][@href="https://example.com/?code=extern%20crate%20foo%3B%0Afn%20main()%20%7B%0Ause%20foo%3A%3Adummy%3B%0Adummy()%3B%0A%7D"]' "Run"
