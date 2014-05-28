// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! HTML fmt support
//!
//! This module contains helper structs for HTML escaping/unescaping with
//! format strings.

use std::fmt;
use escape::{EscapeWriter, UnescapeWriter, EscapeDefault};

/// Wrapper struct which will emit the HTML-escaped version of the contained
/// string when passed to a format string.
///
/// Escapes using the semantics of `html::escape::EscapeDefault`.
pub struct Escape<T>(pub T);

impl<T: fmt::Show> fmt::Show for Escape<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let Escape(ref inner) = *self;
        (write!(EscapeWriter::new(fmt.by_ref(), EscapeDefault), "{}", inner))
               .map_err(|_| fmt::WriteError)
    }
}

/// Wrapper struct which will unescape HTML entities in the contained
/// string when passed to a format string.
pub struct Unescape<T>(pub T);

impl<T: fmt::Show> fmt::Show for Unescape<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let Unescape(ref inner) = *self;
        let mut w = UnescapeWriter::new(fmt.by_ref());
        (write!(w, "{}", inner)).and_then(|_| w.flush_outer()).map_err(|_| fmt::WriteError)
    }
}
