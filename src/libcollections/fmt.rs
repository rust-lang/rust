// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Formatting support for `String`.
//!
//! See `core::fmt` and `std::fmt` for full documentation on string
//! formatting.

#![stable(feature = "rust1", since = "1.0.0")]

use core::fmt;

use string;

/// The format function takes a precompiled format string and a list of
/// arguments, to return the resulting formatted string.
///
/// # Arguments
///
///   * args - a structure of arguments generated via the `format_args!` macro.
///
/// # Example
///
/// ```rust
/// use std::fmt;
///
/// let s = fmt::format(format_args!("Hello, {}!", "world"));
/// assert_eq!(s, "Hello, world!".to_string());
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn format(args: fmt::Arguments) -> string::String {
    // FIXME #21826
    use core::fmt::Writer;
    let mut output = string::String::new();
    let _ = write!(&mut output, "{}", args);
    output
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use fmt;

    #[test]
    fn test_format() {
        let s = fmt::format(format_args!("Hello, {}!", "world"));
        assert_eq!(s.as_slice(), "Hello, world!");
    }
}
