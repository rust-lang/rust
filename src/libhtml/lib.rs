// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! HTML Escaping
//!
//! # Examples
//!
//! Escaping some HTML text:
//!
//! ```rust
//! use html;
//!
//! fn main() {
//!     let original = "<p>Dr. Jekyll & Mr. Hyde</p>";
//!     let escaped = html::escape(original);
//!     assert_eq!(escaped.as_slice(), "&lt;p&gt;Dr. Jekyll &amp; Mr. Hyde&lt;p&gt;");
//!     let unescaped = html::unescape(escaped);
//!     assert_eq!(unescaped.as_slice(), original);
//! }
//! ```
//!
//! Or, if you are formating multiple strings, using `html::fmt::Escape` or
//! `html::fmt::Unescape` can be used to reduce allocations, increasing perfomance.
//!
//! ```rust
//! use html::fmt::Escape;
//!
//! fn main() {
//!     println!("<h1>{}</h1><h2>{}</h2>", Escape("<html>"), Escape("in <Rust>"));
//! }
//! ```
//!
//! Finally, `html::escape` has two `Writer` adaptors, `html::escape::EscapeWriter`
//! and `html::escape::UnescapeWriter` that can be used as desired.
//!
//! ```rust
//! use html::escape::UnescapeWriter;
//! use std::io;
//!
//! fn main() {
//!     let mut w = UnescapeWriter::new(io::stdout());
//!     let _ = io::copy(&mut io::stdin(), &mut w);
//! }
//! ```

#![crate_id = "html#0.11-pre"]
#![license = "MIT/ASL2"]
#![crate_type = "dylib"]
#![crate_type = "rlib"]

#![feature(macro_rules)] // used for tests

use std::fmt::Show;
use fmt::{Escape, Unescape};

pub mod escape;
pub mod fmt;
mod entity;

/// Returns a new string with special characters escaped as HTML entities.
///
/// This will escape only 5 characters: `<`, `>`, `&`, `'`, and `"`.
/// `unescape(escape(s)) == s` is always true, but the converse isn't necessarily true.
pub fn escape<T: Show>(s: T) -> ~str {
    format!("{}", Escape(s))
}

/// Returns a new string with HTML entities transformed to unicode characters.
///
/// It escapes a larger range of entities than `escape` escapes. For example,
/// `&aacute;` unescapes to "?", as does `&#225;` and `&xE1;`.
/// `unescape(escape(s)) == s` is always true, but the converse isn't necessarily true.
pub fn unescape<T: Show>(s: T) -> ~str {
    format!("{}", Unescape(s))
}

#[cfg(test)]
mod tests;
