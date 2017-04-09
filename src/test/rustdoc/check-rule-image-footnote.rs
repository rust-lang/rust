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

// ignore-tidy-linelength

// @has foo/fn.f.html
// @has - '<p>markdown test</p>'
// @has - '<p>this is a <a href="https://example.com" title="this is a title">link</a>.</p>'
// @has - '<hr />'
// @has - '<p>a footnote<sup id="supref1"><a href="#ref1">1</a></sup>.</p>'
// @has - '<p>another footnote<sup id="supref2"><a href="#ref2">2</a></sup>.</p>'
// @has - '<p><img src="https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png" alt="Rust" /></p>'
// @has - '<div class="footnotes"><hr><ol><li id="ref1">'
// @has - '<p>Thing&nbsp;<a href="#supref1" rev="footnote">↩</a></p></li><li id="ref2">'
// @has - '<p>Another Thing&nbsp;<a href="#supref2" rev="footnote">↩</a></p></li></ol></div>'
/// markdown test
///
/// this is a [link].
///
/// [link]: https://example.com "this is a title"
///
/// -----------
///
/// a footnote[^footnote].
///
/// another footnote[^footnotebis].
///
/// [^footnote]: Thing
///
///
/// [^footnotebis]: Another Thing
///
///
/// ![Rust](https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png)
pub fn f() {}
