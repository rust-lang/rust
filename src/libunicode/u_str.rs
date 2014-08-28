// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15679

/*!
 * Unicode-intensive string manipulations.
 *
 * This module provides functionality to `str` that requires the Unicode
 * methods provided by the UnicodeChar trait.
 */

use core::clone::Clone;
use core::cmp;
use core::collections::Collection;
use core::iter::{Filter, AdditiveIterator, Iterator, DoubleEndedIterator};
use core::option::{Option, None, Some};
use core::str::{CharSplits, StrSlice};
use u_char;
use u_char::UnicodeChar;
use tables::grapheme::GraphemeCat;

/// An iterator over the words of a string, separated by a sequence of whitespace
pub type Words<'a> =
    Filter<'a, &'a str, CharSplits<'a, extern "Rust" fn(char) -> bool>>;

/// Methods for Unicode string slices
pub trait UnicodeStrSlice<'a> {
    /// Returns an iterator over the
    /// [grapheme clusters](http://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    /// of the string.
    ///
    /// If `is_extended` is true, the iterator is over the *extended grapheme clusters*;
    /// otherwise, the iterator is over the *legacy grapheme clusters*.
    /// [UAX#29](http://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    /// recommends extended grapheme cluster boundaries for general processing.
    ///
    /// # Example
    ///
    /// ```rust
    /// let gr1 = "a\u0310e\u0301o\u0308\u0332".graphemes(true).collect::<Vec<&str>>();
    /// let b: &[_] = &["a\u0310", "e\u0301", "o\u0308\u0332"];
    /// assert_eq!(gr1.as_slice(), b);
    /// let gr2 = "a\r\nb🇷🇺🇸🇹".graphemes(true).collect::<Vec<&str>>();
    /// let b: &[_] = &["a", "\r\n", "b", "🇷🇺🇸🇹"];
    /// assert_eq!(gr2.as_slice(), b);
    /// ```
    fn graphemes(&self, is_extended: bool) -> Graphemes<'a>;

    /// Returns an iterator over the grapheme clusters of self and their byte offsets.
    /// See `graphemes()` method for more information.
    ///
    /// # Example
    ///
    /// ```rust
    /// let gr_inds = "a̐éö̲\r\n".grapheme_indices(true).collect::<Vec<(uint, &str)>>();
    /// let b: &[_] = &[(0u, "a̐"), (3, "é"), (6, "ö̲"), (11, "\r\n")];
    /// assert_eq!(gr_inds.as_slice(), b);
    /// ```
    fn grapheme_indices(&self, is_extended: bool) -> GraphemeIndices<'a>;

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
    fn width(&self, is_cjk: bool) -> uint;

    /// Returns a string with leading and trailing whitespace removed.
    fn trim(&self) -> &'a str;

    /// Returns a string with leading whitespace removed.
    fn trim_left(&self) -> &'a str;

    /// Returns a string with trailing whitespace removed.
    fn trim_right(&self) -> &'a str;
}

impl<'a> UnicodeStrSlice<'a> for &'a str {
    #[inline]
    fn graphemes(&self, is_extended: bool) -> Graphemes<'a> {
        Graphemes { string: *self, extended: is_extended, cat: None, catb: None }
    }

    #[inline]
    fn grapheme_indices(&self, is_extended: bool) -> GraphemeIndices<'a> {
        GraphemeIndices { start_offset: self.as_ptr() as uint, iter: self.graphemes(is_extended) }
    }

    #[inline]
    fn words(&self) -> Words<'a> {
        self.split(u_char::is_whitespace).filter(|s| !s.is_empty())
    }

    #[inline]
    fn is_whitespace(&self) -> bool { self.chars().all(u_char::is_whitespace) }

    #[inline]
    fn is_alphanumeric(&self) -> bool { self.chars().all(u_char::is_alphanumeric) }

    #[inline]
    fn width(&self, is_cjk: bool) -> uint {
        self.chars().map(|c| c.width(is_cjk).unwrap_or(0)).sum()
    }

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

/// External iterator for grapheme clusters and byte offsets.
#[deriving(Clone)]
pub struct GraphemeIndices<'a> {
    start_offset: uint,
    iter: Graphemes<'a>,
}

impl<'a> Iterator<(uint, &'a str)> for GraphemeIndices<'a> {
    #[inline]
    fn next(&mut self) -> Option<(uint, &'a str)> {
        self.iter.next().map(|s| (s.as_ptr() as uint - self.start_offset, s))
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

impl<'a> DoubleEndedIterator<(uint, &'a str)> for GraphemeIndices<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<(uint, &'a str)> {
        self.iter.next_back().map(|s| (s.as_ptr() as uint - self.start_offset, s))
    }
}

/// External iterator for a string's
/// [grapheme clusters](http://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries).
#[deriving(Clone)]
pub struct Graphemes<'a> {
    string: &'a str,
    extended: bool,
    cat: Option<GraphemeCat>,
    catb: Option<GraphemeCat>,
}

// state machine for cluster boundary rules
#[deriving(PartialEq,Eq)]
enum GraphemeState {
    Start,
    FindExtend,
    HangulL,
    HangulLV,
    HangulLVT,
    Regional,
}

impl<'a> Iterator<&'a str> for Graphemes<'a> {
    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let slen = self.string.len();
        (cmp::min(slen, 1u), Some(slen))
    }

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        use tables::grapheme as gr;
        if self.string.len() == 0 {
            return None;
        }

        let mut take_curr = true;
        let mut idx = 0;
        let mut state = Start;
        let mut cat = gr::GC_Any;
        for (curr, ch) in self.string.char_indices() {
            idx = curr;

            // retrieve cached category, if any
            // We do this because most of the time we would end up
            // looking up each character twice.
            cat = match self.cat {
                None => gr::grapheme_category(ch),
                _ => self.cat.take().unwrap()
            };

            if match cat {
                gr::GC_Extend => true,
                gr::GC_SpacingMark if self.extended => true,
                _ => false
            } {
                    state = FindExtend;     // rule GB9/GB9a
                    continue;
            }

            state = match state {
                Start if '\r' == ch => {
                    let slen = self.string.len();
                    let nidx = idx + 1;
                    if nidx != slen && self.string.char_at(nidx) == '\n' {
                        idx = nidx;             // rule GB3
                    }
                    break;                      // rule GB4
                }
                Start => match cat {
                    gr::GC_Control => break,
                    gr::GC_L => HangulL,
                    gr::GC_LV | gr::GC_V => HangulLV,
                    gr::GC_LVT | gr::GC_T => HangulLVT,
                    gr::GC_RegionalIndicator => Regional,
                    _ => FindExtend
                },
                FindExtend => {         // found non-extending when looking for extending
                    take_curr = false;
                    break;
                },
                HangulL => match cat {      // rule GB6: L x (L|V|LV|LVT)
                    gr::GC_L => continue,
                    gr::GC_LV | gr::GC_V => HangulLV,
                    gr::GC_LVT => HangulLVT,
                    _ => {
                        take_curr = false;
                        break;
                    }
                },
                HangulLV => match cat {     // rule GB7: (LV|V) x (V|T)
                    gr::GC_V => continue,
                    gr::GC_T => HangulLVT,
                    _ => {
                        take_curr = false;
                        break;
                    }
                },
                HangulLVT => match cat {    // rule GB8: (LVT|T) x T
                    gr::GC_T => continue,
                    _ => {
                        take_curr = false;
                        break;
                    }
                },
                Regional => match cat {     // rule GB8a
                    gr::GC_RegionalIndicator => continue,
                    _ => {
                        take_curr = false;
                        break;
                    }
                }
            }
        }

        self.cat = if take_curr {
            idx = self.string.char_range_at(idx).next;
            None
        } else {
            Some(cat)
        };

        let retstr = self.string.slice_to(idx);
        self.string = self.string.slice_from(idx);
        Some(retstr)
    }
}

impl<'a> DoubleEndedIterator<&'a str> for Graphemes<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> {
        use tables::grapheme as gr;
        if self.string.len() == 0 {
            return None;
        }

        let mut take_curr = true;
        let mut idx = self.string.len();
        let mut previdx = idx;
        let mut state = Start;
        let mut cat = gr::GC_Any;
        for (curr, ch) in self.string.char_indices().rev() {
            previdx = idx;
            idx = curr;

            // cached category, if any
            cat = match self.catb {
                None => gr::grapheme_category(ch),
                _ => self.catb.take().unwrap()
            };

            // a matching state machine that runs *backwards* across an input string
            // note that this has some implications for the Hangul matching, since
            // we now need to know what the rightward letter is:
            //
            // Right to left, we have:
            //      L x L
            //      V x (L|V|LV)
            //      T x (V|T|LV|LVT)
            // HangulL means the letter to the right is L
            // HangulLV means the letter to the right is V
            // HangulLVT means the letter to the right is T
            state = match state {
                Start if '\n' == ch => {
                    if idx > 0 && '\r' == self.string.char_at_reverse(idx) {
                        idx -= 1;       // rule GB3
                    }
                    break;              // rule GB4
                },
                Start | FindExtend => match cat {
                    gr::GC_Extend => FindExtend,
                    gr::GC_SpacingMark if self.extended => FindExtend,
                    gr::GC_L | gr::GC_LV | gr::GC_LVT => HangulL,
                    gr::GC_V => HangulLV,
                    gr::GC_T => HangulLVT,
                    gr::GC_RegionalIndicator => Regional,
                    gr::GC_Control => {
                        take_curr = Start == state;
                        break;
                    },
                    _ => break
                },
                HangulL => match cat {      // char to right is an L
                    gr::GC_L => continue,               // L x L is the only legal match
                    _ => {
                        take_curr = false;
                        break;
                    }
                },
                HangulLV => match cat {     // char to right is a V
                    gr::GC_V => continue,               // V x V, right char is still V
                    gr::GC_L | gr::GC_LV => HangulL,    // (L|V) x V, right char is now L
                    _ => {
                        take_curr = false;
                        break;
                    }
                },
                HangulLVT => match cat {    // char to right is a T
                    gr::GC_T => continue,               // T x T, right char is still T
                    gr::GC_V => HangulLV,               // V x T, right char is now V
                    gr::GC_LV | gr::GC_LVT => HangulL,  // (LV|LVT) x T, right char is now L
                    _ => {
                        take_curr = false;
                        break;
                    }
                },
                Regional => match cat {     // rule GB8a
                    gr::GC_RegionalIndicator => continue,
                    _ => {
                        take_curr = false;
                        break;
                    }
                }
            }
        }

        self.catb = if take_curr {
            None
        } else  {
            idx = previdx;
            Some(cat)
        };

        let retstr = self.string.slice_from(idx);
        self.string = self.string.slice_to(idx);
        Some(retstr)
    }
}
