// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

// compile-flags:-Z unstable-options --extern-html-root-url core=https://example.com/core/0.1.0

// @has extern_html_root_url/index.html
// @has - '//a/@href' 'https://example.com/core/0.1.0/core/iter/index.html'
#[doc(no_inline)]
pub use std::iter;
