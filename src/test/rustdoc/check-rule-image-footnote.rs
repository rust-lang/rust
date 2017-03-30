// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "foo"]

// @has foo/fn.f.html
// @has - '<p>hard break: after hard break</p><hr>'
// @has - '<img src="https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png" alt="Rust">'
// @has - '<li id="ref1">'
// @has - '<sup id="supref1"><a href="#ref1">1</a></sup>'
/// markdown test
///
/// this is a [link].
///
/// [link]: https://example.com "this is a title"
///
/// hard break:
/// after hard break
///
/// -----------
///
/// a footnote[^footnote].
///
/// [^footnote]: Thing
///
/// ![Rust](https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png)
#[deprecated(note = "Struct<T>")]
pub fn f() {}
