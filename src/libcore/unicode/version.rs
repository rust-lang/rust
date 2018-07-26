// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Represents a Unicode Version.
///
/// See also: <http://www.unicode.org/versions/>
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[unstable(feature = "unicode_version", issue = "49726")]
pub struct UnicodeVersion {
    /// Major version.
    pub major: u32,

    /// Minor version.
    pub minor: u32,

    /// Micro (or Update) version.
    pub micro: u32,

    // Private field to keep struct expandable.
    pub(crate) _priv: (),
}
