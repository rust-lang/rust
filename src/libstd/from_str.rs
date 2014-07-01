// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The `FromStr` trait for types that can be created from strings

#![experimental]

use option::{Option, Some, None};
use string::String;
use str::StrAllocating;

/// A trait to abstract the idea of creating a new instance of a type from a
/// string.
#[experimental = "might need to return Result"]
pub trait FromStr {
    /// Parses a string `s` to return an optional value of this type. If the
    /// string is ill-formatted, the None is returned.
    fn from_str(s: &str) -> Option<Self>;
}

/// A utility function that just calls FromStr::from_str
pub fn from_str<A: FromStr>(s: &str) -> Option<A> {
    FromStr::from_str(s)
}

impl FromStr for bool {
    /// Parse a `bool` from a string.
    ///
    /// Yields an `Option<bool>`, because `s` may or may not actually be parseable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!(from_str::<bool>("true"), Some(true));
    /// assert_eq!(from_str::<bool>("false"), Some(false));
    /// assert_eq!(from_str::<bool>("not even a boolean"), None);
    /// ```
    #[inline]
    fn from_str(s: &str) -> Option<bool> {
        match s {
            "true"  => Some(true),
            "false" => Some(false),
            _       => None,
        }
    }
}

impl FromStr for String {
    #[inline]
    fn from_str(s: &str) -> Option<String> {
        Some(s.to_string())
    }
}

#[cfg(test)]
mod test {
    use prelude::*;

    #[test]
    fn test_bool_from_str() {
        assert_eq!(from_str::<bool>("true"), Some(true));
        assert_eq!(from_str::<bool>("false"), Some(false));
        assert_eq!(from_str::<bool>("not even a boolean"), None);
    }
}
