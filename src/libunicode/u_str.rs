// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Unicode-intensive string manipulations.
 *
 * This module provides functionality to `str` that requires the Unicode
 * methods provided by the UnicodeChar trait.
 */

use core::collections::Collection;
use core::iter::{Filter};
use core::str::{CharSplits, StrSlice};
use core::iter::Iterator;
use u_char;

/// An iterator over the words of a string, separated by a sequence of whitespace
pub type Words<'a> =
    Filter<'a, &'a str, CharSplits<'a, extern "Rust" fn(char) -> bool>>;

/// Methods for Unicode string slices
pub trait UnicodeStrSlice<'a> {
    /// An iterator over the words of a string (subsequences separated
    /// by any sequence of whitespace). Sequences of whitespace are
    /// collapsed, so empty "words" are not included.
    ///
    /// # Example
    ///
    /// ```rust
    /// let some_words = " Mary   had\ta little  \n\t lamb";
    /// let v: Vec<&str> = some_words.words().collect();
    /// assert_eq!(v, vec!["Mary", "had", "a", "little", "lamb"]);
    /// ```
    fn words(&self) -> Words<'a>;

    /// Returns true if the string contains only whitespace.
    ///
    /// Whitespace characters are determined by `char::is_whitespace`.
    ///
    /// # Example
    ///
    /// ```rust
    /// assert!(" \t\n".is_whitespace());
    /// assert!("".is_whitespace());
    ///
    /// assert!( !"abc".is_whitespace());
    /// ```
    fn is_whitespace(&self) -> bool;

    /// Returns true if the string contains only alphanumeric code
    /// points.
    ///
    /// Alphanumeric characters are determined by `char::is_alphanumeric`.
    ///
    /// # Example
    ///
    /// ```rust
    /// assert!("Löwe老虎Léopard123".is_alphanumeric());
    /// assert!("".is_alphanumeric());
    ///
    /// assert!( !" &*~".is_alphanumeric());
    /// ```
    fn is_alphanumeric(&self) -> bool;

    /// Returns a string's displayed width in columns, treating control
    /// characters as zero-width.
    ///
    /// `is_cjk` determines behavior for characters in the Ambiguous category:
    /// if `is_cjk` is `true`, these are 2 columns wide; otherwise, they are 1.
    /// In CJK locales, `is_cjk` should be `true`, else it should be `false`.
    /// [Unicode Standard Annex #11](http://www.unicode.org/reports/tr11/)
    /// recommends that these characters be treated as 1 column (i.e.,
    /// `is_cjk` = `false`) if the locale is unknown.
    //fn width(&self, is_cjk: bool) -> uint;

    /// Returns a string with leading and trailing whitespace removed.
    fn trim(&self) -> &'a str;

    /// Returns a string with leading whitespace removed.
    fn trim_left(&self) -> &'a str;

    /// Returns a string with trailing whitespace removed.
    fn trim_right(&self) -> &'a str;
}

impl<'a> UnicodeStrSlice<'a> for &'a str {
    #[inline]
    fn words(&self) -> Words<'a> {
        self.split(u_char::is_whitespace).filter(|s| !s.is_empty())
    }

    #[inline]
    fn is_whitespace(&self) -> bool { self.chars().all(u_char::is_whitespace) }

    #[inline]
    fn is_alphanumeric(&self) -> bool { self.chars().all(u_char::is_alphanumeric) }

    #[inline]
    fn trim(&self) -> &'a str {
        self.trim_left().trim_right()
    }

    #[inline]
    fn trim_left(&self) -> &'a str {
        self.trim_left_chars(u_char::is_whitespace)
    }

    #[inline]
    fn trim_right(&self) -> &'a str {
        self.trim_right_chars(u_char::is_whitespace)
    }
}
