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

//! Unicode-intensive string manipulations.
//!
//! This module provides functionality to `str` that requires the Unicode methods provided by the
//! unicode parts of the CharExt trait.

use core::prelude::*;

use core::char;
use core::cmp;
use core::iter::Filter;
use core::mem;
use core::slice;
use core::str::Split;

use tables::grapheme::GraphemeCat;
use tables::word::WordCat;

/// An iterator over the words of a string, separated by a sequence of whitespace
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Words<'a> {
    inner: Filter<Split<'a, fn(char) -> bool>, fn(&&str) -> bool>,
}

/// Methods for Unicode string slices
#[allow(missing_docs)] // docs in libcollections
pub trait UnicodeStr {
    fn graphemes<'a>(&'a self, is_extended: bool) -> Graphemes<'a>;
    fn grapheme_indices<'a>(&'a self, is_extended: bool) -> GraphemeIndices<'a>;
    fn words<'a>(&'a self) -> Words<'a>;
    fn words_unicode<'a>(&'a self) -> UnicodeWords<'a>;
    fn is_whitespace(&self) -> bool;
    fn is_alphanumeric(&self) -> bool;
    fn width(&self, is_cjk: bool) -> usize;
    fn trim<'a>(&'a self) -> &'a str;
    fn trim_left<'a>(&'a self) -> &'a str;
    fn trim_right<'a>(&'a self) -> &'a str;
    fn split_words_uax29<'a>(&'a self) -> UWordBounds<'a>;
    fn split_words_uax29_indices<'a>(&'a self) -> UWordBoundIndices<'a>;
}

impl UnicodeStr for str {
    #[inline]
    fn graphemes(&self, is_extended: bool) -> Graphemes {
        Graphemes { string: self, extended: is_extended, cat: None, catb: None }
    }

    #[inline]
    fn grapheme_indices(&self, is_extended: bool) -> GraphemeIndices {
        GraphemeIndices { start_offset: self.as_ptr() as usize, iter: self.graphemes(is_extended) }
    }

    #[inline]
    fn words(&self) -> Words {
        fn is_not_empty(s: &&str) -> bool { !s.is_empty() }
        let is_not_empty: fn(&&str) -> bool = is_not_empty; // coerce to fn pointer

        fn is_whitespace(c: char) -> bool { c.is_whitespace() }
        let is_whitespace: fn(char) -> bool = is_whitespace; // coerce to fn pointer

        Words { inner: self.split(is_whitespace).filter(is_not_empty) }
    }

    #[inline]
    fn words_unicode(&self) -> UnicodeWords {
        fn has_alphanumeric(s: &&str) -> bool { s.chars().any(|c| c.is_alphanumeric()) }
        let has_alphanumeric: fn(&&str) -> bool = has_alphanumeric; // coerce to fn pointer

        UnicodeWords { inner: self.split_words_uax29().filter(has_alphanumeric) }
    }

    #[inline]
    fn is_whitespace(&self) -> bool { self.chars().all(|c| c.is_whitespace()) }

    #[inline]
    fn is_alphanumeric(&self) -> bool { self.chars().all(|c| c.is_alphanumeric()) }

    #[inline]
    fn width(&self, is_cjk: bool) -> usize {
        self.chars().map(|c| c.width(is_cjk).unwrap_or(0)).sum()
    }

    #[inline]
    fn trim(&self) -> &str {
        self.trim_matches(|c: char| c.is_whitespace())
    }

    #[inline]
    fn trim_left(&self) -> &str {
        self.trim_left_matches(|c: char| c.is_whitespace())
    }

    #[inline]
    fn trim_right(&self) -> &str {
        self.trim_right_matches(|c: char| c.is_whitespace())
    }

    #[inline]
    fn split_words_uax29(&self) -> UWordBounds {
        UWordBounds { string: self, cat: None, catb: None }
    }

    #[inline]
    fn split_words_uax29_indices(&self) -> UWordBoundIndices {
        UWordBoundIndices { start_offset: self.as_ptr() as usize, iter: self.split_words_uax29() }
    }
}

/// External iterator for grapheme clusters and byte offsets.
#[derive(Clone)]
pub struct GraphemeIndices<'a> {
    start_offset: usize,
    iter: Graphemes<'a>,
}

impl<'a> Iterator for GraphemeIndices<'a> {
    type Item = (usize, &'a str);

    #[inline]
    fn next(&mut self) -> Option<(usize, &'a str)> {
        self.iter.next().map(|s| (s.as_ptr() as usize - self.start_offset, s))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a> DoubleEndedIterator for GraphemeIndices<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<(usize, &'a str)> {
        self.iter.next_back().map(|s| (s.as_ptr() as usize - self.start_offset, s))
    }
}

/// External iterator for a string's
/// [grapheme clusters](http://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries).
#[derive(Clone)]
pub struct Graphemes<'a> {
    string: &'a str,
    extended: bool,
    cat: Option<GraphemeCat>,
    catb: Option<GraphemeCat>,
}

// state machine for cluster boundary rules
#[derive(PartialEq,Eq)]
enum GraphemeState {
    Start,
    FindExtend,
    HangulL,
    HangulLV,
    HangulLVT,
    Regional,
}

impl<'a> Iterator for Graphemes<'a> {
    type Item = &'a str;

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let slen = self.string.len();
        (cmp::min(slen, 1), Some(slen))
    }

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        use self::GraphemeState::*;
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
                    gr::GC_Regional_Indicator => Regional,
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
                    gr::GC_Regional_Indicator => continue,
                    _ => {
                        take_curr = false;
                        break;
                    }
                }
            }
        }

        self.cat = if take_curr {
            idx = idx + self.string.char_at(idx).len_utf8();
            None
        } else {
            Some(cat)
        };

        let retstr = &self.string[..idx];
        self.string = &self.string[idx..];
        Some(retstr)
    }
}

impl<'a> DoubleEndedIterator for Graphemes<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> {
        use self::GraphemeState::*;
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
                    gr::GC_Regional_Indicator => Regional,
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
                    gr::GC_Regional_Indicator => continue,
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

        let retstr = &self.string[idx..];
        self.string = &self.string[..idx];
        Some(retstr)
    }
}

// https://tools.ietf.org/html/rfc3629
static UTF8_CHAR_WIDTH: [u8; 256] = [
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0x1F
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0x3F
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0x5F
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0x7F
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 0x9F
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 0xBF
0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, // 0xDF
3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3, // 0xEF
4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0, // 0xFF
];

/// Given a first byte, determine how many bytes are in this UTF-8 character
#[inline]
pub fn utf8_char_width(b: u8) -> usize {
    return UTF8_CHAR_WIDTH[b as usize] as usize;
}

/// Determines if a vector of `u16` contains valid UTF-16
pub fn is_utf16(v: &[u16]) -> bool {
    let mut it = v.iter();
    macro_rules! next { ($ret:expr) => {
            match it.next() { Some(u) => *u, None => return $ret }
        }
    }
    loop {
        let u = next!(true);

        match char::from_u32(u as u32) {
            Some(_) => {}
            None => {
                let u2 = next!(false);
                if u < 0xD7FF || u > 0xDBFF ||
                    u2 < 0xDC00 || u2 > 0xDFFF { return false; }
            }
        }
    }
}

/// An iterator that decodes UTF-16 encoded codepoints from a vector
/// of `u16`s.
#[derive(Clone)]
pub struct Utf16Items<'a> {
    iter: slice::Iter<'a, u16>
}
/// The possibilities for values decoded from a `u16` stream.
#[derive(Copy, PartialEq, Eq, Clone, Debug)]
pub enum Utf16Item {
    /// A valid codepoint.
    ScalarValue(char),
    /// An invalid surrogate without its pair.
    LoneSurrogate(u16)
}

impl Utf16Item {
    /// Convert `self` to a `char`, taking `LoneSurrogate`s to the
    /// replacement character (U+FFFD).
    #[inline]
    pub fn to_char_lossy(&self) -> char {
        match *self {
            Utf16Item::ScalarValue(c) => c,
            Utf16Item::LoneSurrogate(_) => '\u{FFFD}'
        }
    }
}

impl<'a> Iterator for Utf16Items<'a> {
    type Item = Utf16Item;

    fn next(&mut self) -> Option<Utf16Item> {
        let u = match self.iter.next() {
            Some(u) => *u,
            None => return None
        };

        if u < 0xD800 || 0xDFFF < u {
            // not a surrogate
            Some(Utf16Item::ScalarValue(unsafe {mem::transmute(u as u32)}))
        } else if u >= 0xDC00 {
            // a trailing surrogate
            Some(Utf16Item::LoneSurrogate(u))
        } else {
            // preserve state for rewinding.
            let old = self.iter.clone();

            let u2 = match self.iter.next() {
                Some(u2) => *u2,
                // eof
                None => return Some(Utf16Item::LoneSurrogate(u))
            };
            if u2 < 0xDC00 || u2 > 0xDFFF {
                // not a trailing surrogate so we're not a valid
                // surrogate pair, so rewind to redecode u2 next time.
                self.iter = old.clone();
                return Some(Utf16Item::LoneSurrogate(u))
            }

            // all ok, so lets decode it.
            let c = (((u - 0xD800) as u32) << 10 | (u2 - 0xDC00) as u32) + 0x1_0000;
            Some(Utf16Item::ScalarValue(unsafe {mem::transmute(c)}))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.iter.size_hint();
        // we could be entirely valid surrogates (2 elements per
        // char), or entirely non-surrogates (1 element per char)
        (low / 2, high)
    }
}

/// Create an iterator over the UTF-16 encoded codepoints in `v`,
/// returning invalid surrogates as `LoneSurrogate`s.
///
/// # Examples
///
/// ```
/// # #![feature(unicode)]
/// extern crate unicode;
///
/// use unicode::str::Utf16Item::{ScalarValue, LoneSurrogate};
///
/// fn main() {
///     // 𝄞mus<invalid>ic<invalid>
///     let v = [0xD834, 0xDD1E, 0x006d, 0x0075,
///              0x0073, 0xDD1E, 0x0069, 0x0063,
///              0xD834];
///
///     assert_eq!(unicode::str::utf16_items(&v).collect::<Vec<_>>(),
///                vec![ScalarValue('𝄞'),
///                     ScalarValue('m'), ScalarValue('u'), ScalarValue('s'),
///                     LoneSurrogate(0xDD1E),
///                     ScalarValue('i'), ScalarValue('c'),
///                     LoneSurrogate(0xD834)]);
/// }
/// ```
pub fn utf16_items<'a>(v: &'a [u16]) -> Utf16Items<'a> {
    Utf16Items { iter : v.iter() }
}

/// Iterator adaptor for encoding `char`s to UTF-16.
#[derive(Clone)]
pub struct Utf16Encoder<I> {
    chars: I,
    extra: u16
}

impl<I> Utf16Encoder<I> {
    /// Create an UTF-16 encoder from any `char` iterator.
    pub fn new(chars: I) -> Utf16Encoder<I> where I: Iterator<Item=char> {
        Utf16Encoder { chars: chars, extra: 0 }
    }
}

impl<I> Iterator for Utf16Encoder<I> where I: Iterator<Item=char> {
    type Item = u16;

    #[inline]
    fn next(&mut self) -> Option<u16> {
        if self.extra != 0 {
            let tmp = self.extra;
            self.extra = 0;
            return Some(tmp);
        }

        let mut buf = [0; 2];
        self.chars.next().map(|ch| {
            let n = CharExt::encode_utf16(ch, &mut buf).unwrap_or(0);
            if n == 2 { self.extra = buf[1]; }
            buf[0]
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.chars.size_hint();
        // every char gets either one u16 or two u16,
        // so this iterator is between 1 or 2 times as
        // long as the underlying iterator.
        (low, high.and_then(|n| n.checked_mul(2)))
    }
}

impl<'a> Iterator for Words<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> { self.inner.next() }
}
impl<'a> DoubleEndedIterator for Words<'a> {
    fn next_back(&mut self) -> Option<&'a str> { self.inner.next_back() }
}

/// An iterator over the substrings of a string which, after splitting the string on
/// [word boundaries](http://www.unicode.org/reports/tr29/#Word_Boundaries),
/// contain any characters with the
/// [Alphabetic](http://unicode.org/reports/tr44/#Alphabetic)
/// property, or with
/// [General_Category=Number](http://unicode.org/reports/tr44/#General_Category_Values).
pub struct UnicodeWords<'a> {
    inner: Filter<UWordBounds<'a>, fn(&&str) -> bool>,
}

impl<'a> Iterator for UnicodeWords<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> { self.inner.next() }
}
impl<'a> DoubleEndedIterator for UnicodeWords<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> { self.inner.next_back() }
}

/// External iterator for a string's
/// [word boundaries](http://www.unicode.org/reports/tr29/#Word_Boundaries).
#[derive(Clone)]
pub struct UWordBounds<'a> {
    string: &'a str,
    cat: Option<WordCat>,
    catb: Option<WordCat>,
}

/// External iterator for word boundaries and byte offsets.
#[derive(Clone)]
pub struct UWordBoundIndices<'a> {
    start_offset: usize,
    iter: UWordBounds<'a>,
}

impl<'a> Iterator for UWordBoundIndices<'a> {
    type Item = (usize, &'a str);

    #[inline]
    fn next(&mut self) -> Option<(usize, &'a str)> {
        self.iter.next().map(|s| (s.as_ptr() as usize - self.start_offset, s))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a> DoubleEndedIterator for UWordBoundIndices<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<(usize, &'a str)> {
        self.iter.next_back().map(|s| (s.as_ptr() as usize - self.start_offset, s))
    }
}

// state machine for word boundary rules
#[derive(Clone,Copy,PartialEq,Eq)]
enum UWordBoundsState {
    Start,
    Letter,
    HLetter,
    Numeric,
    Katakana,
    ExtendNumLet,
    Regional,
    FormatExtend(FormatExtendType),
}

// subtypes for FormatExtend state in UWordBoundsState
#[derive(Clone,Copy,PartialEq,Eq)]
enum FormatExtendType {
    AcceptAny,
    AcceptNone,
    RequireLetter,
    RequireHLetter,
    AcceptQLetter,
    RequireNumeric,
}

impl<'a> Iterator for UWordBounds<'a> {
    type Item = &'a str;

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let slen = self.string.len();
        (cmp::min(slen, 1), Some(slen))
    }

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        use self::UWordBoundsState::*;
        use self::FormatExtendType::*;
        use tables::word as wd;
        if self.string.len() == 0 {
            return None;
        }

        let mut take_curr = true;
        let mut take_cat = true;
        let mut idx = 0;
        let mut saveidx = 0;
        let mut state = Start;
        let mut cat = wd::WC_Any;
        let mut savecat = wd::WC_Any;
        for (curr, ch) in self.string.char_indices() {
            idx = curr;

            // if there's a category cached, grab it
            cat = match self.cat {
                None => wd::word_category(ch),
                _ => self.cat.take().unwrap()
            };
            take_cat = true;

            // handle rule WB4
            // just skip all format and extend chars
            // note that Start is a special case: if there's a bunch of Format | Extend
            // characters at the beginning of a block of text, dump them out as one unit.
            //
            // (This is not obvious from the wording of UAX#29, but if you look at the
            // test cases http://www.unicode.org/Public/UNIDATA/auxiliary/WordBreakTest.txt
            // then the "correct" interpretation of WB4 becomes apparent.)
            if state != Start && (cat == wd::WC_Extend || cat == wd::WC_Format) {
                continue;
            }

            state = match state {
                Start if cat == wd::WC_CR => {
                    idx += match self.get_next_cat(idx) {
                        Some(ncat) if ncat == wd::WC_LF => 1,       // rule WB3
                        _ => 0
                    };
                    break;                                          // rule WB3a
                },
                Start => match cat {
                    wd::WC_ALetter => Letter,           // rule WB5, WB6, WB9, WB13a
                    wd::WC_Hebrew_Letter => HLetter,    // rule WB5, WB6, WB7a, WB7b, WB9, WB13a
                    wd::WC_Numeric => Numeric,          // rule WB8, WB10, WB12, WB13a
                    wd::WC_Katakana => Katakana,        // rule WB13, WB13a
                    wd::WC_ExtendNumLet => ExtendNumLet,    // rule WB13a, WB13b
                    wd::WC_Regional_Indicator => Regional,  // rule WB13c
                    wd::WC_LF | wd::WC_Newline => break,    // rule WB3a
                    _ => {
                        if let Some(ncat) = self.get_next_cat(idx) {                // rule WB4
                            if ncat == wd::WC_Format || ncat == wd::WC_Extend {
                                state = FormatExtend(AcceptNone);
                                self.cat = Some(ncat);
                                continue;
                            }
                        }
                        break;                                                      // rule WB14
                    }
                },
                Letter | HLetter => match cat {
                    wd::WC_ALetter => Letter,                   // rule WB5
                    wd::WC_Hebrew_Letter => HLetter,            // rule WB5
                    wd::WC_Numeric => Numeric,                  // rule WB9
                    wd::WC_ExtendNumLet => ExtendNumLet,        // rule WB13a
                    wd::WC_Double_Quote if state == HLetter => {
                        savecat = cat;
                        saveidx = idx;
                        FormatExtend(RequireHLetter)                        // rule WB7b
                    },
                    wd::WC_Single_Quote if state == HLetter => {
                        FormatExtend(AcceptQLetter)                         // rule WB7a
                    },
                    wd::WC_MidLetter | wd::WC_MidNumLet | wd::WC_Single_Quote => {
                        savecat = cat;
                        saveidx = idx;
                        FormatExtend(RequireLetter)                         // rule WB6
                    },
                    _ => {
                        take_curr = false;
                        break;
                    }
                },
                Numeric => match cat {
                    wd::WC_Numeric => Numeric,                  // rule WB8
                    wd::WC_ALetter => Letter,                   // rule WB10
                    wd::WC_Hebrew_Letter => HLetter,            // rule WB10
                    wd::WC_ExtendNumLet => ExtendNumLet,        // rule WB13a
                    wd::WC_MidNum | wd::WC_MidNumLet | wd::WC_Single_Quote => {
                        savecat = cat;
                        saveidx = idx;
                        FormatExtend(RequireNumeric)            // rule WB12
                    },
                    _ => {
                        take_curr = false;
                        break;
                    }
                },
                Katakana => match cat {
                    wd::WC_Katakana => Katakana,                // rule WB13
                    wd::WC_ExtendNumLet => ExtendNumLet,        // rule WB13a
                    _ => {
                        take_curr = false;
                        break;
                    }
                },
                ExtendNumLet => match cat {
                    wd::WC_ExtendNumLet => ExtendNumLet,        // rule WB13a
                    wd::WC_ALetter => Letter,                   // rule WB13b
                    wd::WC_Hebrew_Letter => HLetter,            // rule WB13b
                    wd::WC_Numeric => Numeric,                  // rule WB13b
                    wd::WC_Katakana => Katakana,                // rule WB13b
                    _ => {
                        take_curr = false;
                        break;
                    }
                },
                Regional => match cat {
                    wd::WC_Regional_Indicator => Regional,      // rule WB13c
                    _ => {
                        take_curr = false;
                        break;
                    }
                },
                FormatExtend(t) => match t {    // handle FormatExtends depending on what type
                    RequireNumeric if cat == wd::WC_Numeric => Numeric,     // rule WB11
                    RequireLetter | AcceptQLetter if cat == wd::WC_ALetter => Letter,   // rule WB7
                    RequireLetter | AcceptQLetter if cat == wd::WC_Hebrew_Letter => HLetter, // WB7a
                    RequireHLetter if cat == wd::WC_Hebrew_Letter => HLetter,   // rule WB7b
                    AcceptNone | AcceptQLetter => {
                        take_curr = false;  // emit all the Format|Extend characters
                        take_cat = false;
                        break;
                    },
                    _ => break      // rewind (in if statement below)
                }
            }
        }

        if let FormatExtend(t) = state {
            // we were looking for something and didn't find it; we have to back up
            if t == RequireLetter || t == RequireHLetter || t == RequireNumeric {
                idx = saveidx;
                cat = savecat;
                take_curr = false;
            }
        }

        self.cat = if take_curr {
            idx = idx + self.string.char_at(idx).len_utf8();
            None
        } else if take_cat {
            Some(cat)
        } else {
            None
        };

        let retstr = &self.string[..idx];
        self.string = &self.string[idx..];
        Some(retstr)
    }
}

impl<'a> DoubleEndedIterator for UWordBounds<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> {
        use self::UWordBoundsState::*;
        use self::FormatExtendType::*;
        use tables::word as wd;
        if self.string.len() == 0 {
            return None;
        }

        let mut take_curr = true;
        let mut take_cat = true;
        let mut idx = self.string.len();
        idx -= self.string.char_at_reverse(idx).len_utf8();
        let mut previdx = idx;
        let mut saveidx = idx;
        let mut state = Start;
        let mut savestate = Start;
        let mut cat = wd::WC_Any;
        for (curr, ch) in self.string.char_indices().rev() {
            previdx = idx;
            idx = curr;

            // if there's a category cached, grab it
            cat = match self.catb {
                None => wd::word_category(ch),
                _ => self.catb.take().unwrap()
            };
            take_cat = true;

            // backward iterator over word boundaries. Mostly the same as the forward
            // iterator, with two weirdnesses:
            // (1) If we encounter a single quote in the Start state, we have to check for a
            //     Hebrew Letter immediately before it.
            // (2) Format and Extend char handling takes some gymnastics.

            if cat == wd::WC_Extend || cat == wd::WC_Format {
                if match state {
                    FormatExtend(_) | Start => false,
                    _ => true
                } {
                    saveidx = previdx;
                    savestate = state;
                    state = FormatExtend(AcceptNone);
                }

                if state != Start {
                    continue;
                }
            } else if state == FormatExtend(AcceptNone) {
                // finished a scan of some Format|Extend chars, restore previous state
                state = savestate;
                previdx = saveidx;
                take_cat = false;
            }

            state = match state {
                Start | FormatExtend(AcceptAny) => match cat {
                    wd::WC_ALetter => Letter,           // rule WB5, WB7, WB10, WB13b
                    wd::WC_Hebrew_Letter => HLetter,    // rule WB5, WB7, WB7c, WB10, WB13b
                    wd::WC_Numeric => Numeric,          // rule WB8, WB9, WB11, WB13b
                    wd::WC_Katakana => Katakana,                    // rule WB13, WB13b
                    wd::WC_ExtendNumLet => ExtendNumLet,                    // rule WB13a
                    wd::WC_Regional_Indicator => Regional,                  // rule WB13c
                    wd::WC_Extend | wd::WC_Format => FormatExtend(AcceptAny),   // rule WB4
                    wd::WC_Single_Quote => {
                        saveidx = idx;
                        FormatExtend(AcceptQLetter)                         // rule WB7a
                    },
                    wd::WC_CR | wd::WC_LF | wd::WC_Newline => {
                        if state == Start {
                            if cat == wd::WC_LF {
                                idx -= match self.get_prev_cat(idx) {
                                    Some(pcat) if pcat == wd::WC_CR => 1,   // rule WB3
                                    _ => 0
                                };
                            }
                        } else {
                            take_curr = false;
                        }
                        break;                                              // rule WB3a
                    },
                    _ => break                              // rule WB14
                },
                Letter | HLetter => match cat {
                    wd::WC_ALetter => Letter,               // rule WB5
                    wd::WC_Hebrew_Letter => HLetter,        // rule WB5
                    wd::WC_Numeric => Numeric,              // rule WB10
                    wd::WC_ExtendNumLet => ExtendNumLet,    // rule WB13b
                    wd::WC_Double_Quote if state == HLetter => {
                        saveidx = previdx;
                        FormatExtend(RequireHLetter)         // rule WB7c
                    },
                    wd::WC_MidLetter | wd::WC_MidNumLet | wd::WC_Single_Quote => {
                        saveidx = previdx;
                        FormatExtend(RequireLetter)          // rule WB7
                    },
                    _ => {
                        take_curr = false;
                        break;
                    }
                },
                Numeric => match cat {
                    wd::WC_Numeric => Numeric,              // rule WB8
                    wd::WC_ALetter => Letter,               // rule WB9
                    wd::WC_Hebrew_Letter => HLetter,        // rule WB9
                    wd::WC_ExtendNumLet => ExtendNumLet,    // rule WB13b
                    wd::WC_MidNum | wd::WC_MidNumLet | wd::WC_Single_Quote => {
                        saveidx = previdx;
                        FormatExtend(RequireNumeric)         // rule WB11
                    },
                    _ => {
                        take_curr = false;
                        break;
                    }
                },
                Katakana => match cat {
                    wd::WC_Katakana => Katakana,            // rule WB13
                    wd::WC_ExtendNumLet => ExtendNumLet,    // rule WB13b
                    _ => {
                        take_curr = false;
                        break;
                    }
                },
                ExtendNumLet => match cat {
                    wd::WC_ExtendNumLet => ExtendNumLet,    // rule WB13a
                    wd::WC_ALetter => Letter,               // rule WB13a
                    wd::WC_Hebrew_Letter => HLetter,        // rule WB13a
                    wd::WC_Numeric => Numeric,              // rule WB13a
                    wd::WC_Katakana => Katakana,            // rule WB13a
                    _ => {
                        take_curr = false;
                        break;
                    }
                },
                Regional => match cat {
                    wd::WC_Regional_Indicator => Regional,  // rule WB13c
                    _ => {
                        take_curr = false;
                        break;
                    }
                },
                FormatExtend(t) => match t {
                    RequireNumeric if cat == wd::WC_Numeric => Numeric,          // rule WB12
                    RequireLetter if cat == wd::WC_ALetter => Letter,            // rule WB6
                    RequireLetter if cat == wd::WC_Hebrew_Letter => HLetter,     // rule WB6
                    AcceptQLetter if cat == wd::WC_Hebrew_Letter => HLetter,     // rule WB7a
                    RequireHLetter if cat == wd::WC_Hebrew_Letter => HLetter,    // rule WB7b
                    _ => break  // backtrack will happens
                }
            }
        }

        if let FormatExtend(t) = state {
            // if we required something but didn't find it, backtrack
            if t == RequireLetter || t == RequireHLetter ||
                t == RequireNumeric || t == AcceptNone || t == AcceptQLetter {
                previdx = saveidx;
                take_cat = false;
                take_curr = false;
            }
        }

        self.catb = if take_curr {
            None
        } else {
            idx = previdx;
            if take_cat {
                Some(cat)
            } else {
                None
            }
        };

        let retstr = &self.string[idx..];
        self.string = &self.string[..idx];
        Some(retstr)
    }
}

impl<'a> UWordBounds<'a> {
    #[inline]
    fn get_next_cat(&self, idx: usize) -> Option<WordCat> {
        use tables::word as wd;
        let nidx = idx + self.string.char_at(idx).len_utf8();
        if nidx < self.string.len() {
            let nch = self.string.char_at(nidx);
            Some(wd::word_category(nch))
        } else {
            None
        }
    }

    #[inline]
    fn get_prev_cat(&self, idx: usize) -> Option<WordCat> {
        use tables::word as wd;
        if idx > 0 {
            let nch = self.string.char_at_reverse(idx);
            Some(wd::word_category(nch))
        } else {
            None
        }
    }
}
