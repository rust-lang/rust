//! Module provides pattern matching features for string-like bytes slice.
//!
//! A ‘string-like bytes slice’ means that types and functions here try to
//! interpret bytes slices as well-formed WTF-8 but don’t assume it is and treat
//! bytes in invalid portions of the slices as characters for the purpose of
//! deciding where character boundaries lie.  This can be demonstrated by how
//! empty pattern is matched (since empty patterns match character boundaries):
//!
//! ```
//! #![feature(pattern, pattern_internals, str_internals)]
//! use core::pattern::{Pattern, Searcher};
//! use core::str_bytes::Bytes;
//!
//! let data = ["Żółw".as_bytes(), &b"\xff\xff\xff"[..], "🕴".as_bytes()].concat();
//! let mut searcher = "".into_searcher(Bytes::from(data.as_slice()));
//! let next = move || searcher.next_match().map(|(x, _)| x);
//! let boundaries = core::iter::from_fn(next).collect::<Vec<_>>();
//! assert_eq!(&[0, 2, 4, 6, 7, 8, 9, 10, 14][..], &boundaries[..]);
//! ```
#![unstable(feature = "str_internals", issue = "none")]

use crate::cmp;
use crate::mem::take;
use crate::ops;
use crate::pattern;
use crate::pattern::{Haystack, MatchOnly, RejectOnly, SearchStep, Searcher};
use crate::str::{try_next_code_point, try_next_code_point_reverse};

type OptRange = Option<(usize, usize)>;
type Range = ops::Range<usize>;

////////////////////////////////////////////////////////////////////////////////
// Bytes wrapper
////////////////////////////////////////////////////////////////////////////////

/// A reference to a string-like bytes slice.
///
/// ‘String-like’ refers to the fact that parts of the data are valid WTF-8 and
/// when we split the slice we don’t want to split well-formed WTF-8 bytes
/// sequences.  This is in a sense a generalisation of a `&str` which allows
/// portions of the buffer to be ill-formed while preserving correctness of
/// existing well-formed parts.
#[derive(Copy, Clone, Debug)]
pub struct Bytes<'a>(&'a [u8]);

impl<'a> Bytes<'a> {
    pub fn as_bytes(self) -> &'a [u8] {
        self.0
    }

    pub fn len(self) -> usize {
        self.0.len()
    }

    pub fn is_empty(self) -> bool {
        self.0.is_empty()
    }

    /// Adjusts range’s start position forward so it points at a potential valid
    /// WTF-8 byte sequence.
    ///
    /// `range` represents a possibly invalid range within the bytes;
    /// furthermore, `range.start` must be non-zero.  This method returns a new
    /// start index which is a valid split position.  If `range` is already
    /// a valid, the method simply returns `range.start`.
    ///
    /// When dealing with ill-formed WTF-8 sequences, this is not guaranteed to
    /// advance position byte at a time.  If you need to be able to advance
    /// position byte at a time use `advance_range_start` instead.
    fn adjust_position_fwd(self, range: Range) -> usize {
        range.start
            + self.as_bytes()[range.clone()]
                .iter()
                .take_while(|chr| !chr.is_utf8_char_boundary())
                .count()
    }

    /// Adjusts position backward so that it points at the closest potential
    /// valid WTF-8 sequence.
    ///
    /// `range` represents a possibly invalid range within the bytes,
    /// furthermore `range.end` must be less that bytes’ length.  This method
    /// returns a new exnd index which is a valid split position.  If `range` is
    /// already a valid, the method simply returns `range.end`.
    ///
    /// When dealing with ill-formed WTF-8 sequences, this is not guaranteed to
    /// advance position byte at a time.  If you need to be able to advance
    /// position character at a time use `advance_range_end` instead.
    fn adjust_position_bwd(self, range: Range) -> usize {
        range.end
            - self.as_bytes()[range.start..range.end + 1]
                .iter()
                .rev()
                .take_while(|chr| !chr.is_utf8_char_boundary())
                .count()
    }

    /// Given a valid range update it’s start so it falls on the next character
    /// boundary.
    ///
    /// `range` must be non-empty.  If it starts with a valid WTF-8 sequence,
    /// this method returns position pass that sequence.  Otherwise, it returns
    /// `range.start + 1`.  In other words, well-formed WTF-8 bytes sequence are
    /// skipped in one go while ill-formed sequences are skipped byte-by-byte.
    fn advance_range_start(self, range: Range) -> usize {
        assert!(!range.is_empty());
        match try_next_code_point(&self.0[range.clone()]) {
            Some((_, len)) => range.start + len,
            None => range.end.min(range.start + 1),
        }
    }

    /// Given a valid range update it’s end so it falls on the previous
    /// character boundary.
    ///
    /// `range` must be non-empty.  If it ends with a valid WTF-8 sequence, this
    /// method returns position of the start of that sequence.  Otherwise, it
    /// returns `range.end - 1`.  In other words, well-formed WTF-8 bytes
    /// sequence are skipped in one go while ill-formed sequences are skipped
    /// byte-by-byte.
    fn advance_range_end(self, range: Range) -> usize {
        assert!(!range.is_empty());
        match try_next_code_point_reverse(&self.0[range.clone()]) {
            Some((_, len)) => range.end - len,
            None => range.end - 1,
        }
    }

    /// Returns valid UTF-8 character at the front of the slice.
    ///
    /// If slice doesn’t start with a valid UTF-8 sequence, returns `None`.
    /// Otherwise returns decoded character and it’s UTF-8 encoding’s length.
    /// WTF-8 sequences which encode surrogates are considered invalid.
    fn get_first_code_point(self) -> Option<(char, usize)> {
        try_next_code_point(&self.0)
    }

    /// Returns valid UTF-8 character at the end of the slice.
    ///
    /// If slice doesn’t end with a valid UTF-8 sequence, returns `None`.
    /// Otherwise returns decoded character and it’s UTF-8 encoding’s length.
    /// WTF-8 sequences which encode surrogates are considered invalid.
    fn get_last_code_point(&self) -> Option<(char, usize)> {
        try_next_code_point_reverse(&self.0)
    }

    /// Looks for the next UTF-8-encoded character in the slice.
    ///
    /// WTF-8 sequences which encode surrogates are considered invalid.
    ///
    /// Returns position of the match, decoded character and UTF-8 length of
    /// that character.
    fn find_code_point_fwd(self, range: Range) -> Option<(usize, char, usize)> {
        let bytes = &self.as_bytes()[range.clone()];
        (0..bytes.len())
            .filter_map(|pos| {
                let (chr, len) = try_next_code_point(&bytes[pos..])?;
                Some((range.start + pos, chr, len))
            })
            .next()
    }

    /// Looks backwards for the next UTF-8 encoded character in the slice.
    ///
    /// WTF-8 sequences which encode surrogates are considered invalid.
    ///
    /// Returns position of the match, decoded character and UTF-8 length of
    /// that character.
    fn find_code_point_bwd(&self, range: Range) -> Option<(usize, char, usize)> {
        let bytes = &self.as_bytes()[range.clone()];
        (0..bytes.len())
            .rev()
            .filter_map(|pos| {
                let (chr, len) = try_next_code_point(&bytes[pos..])?;
                Some((range.start + pos, chr, len))
            })
            .next()
    }
}

impl<'a> From<&'a [u8]> for Bytes<'a> {
    #[inline]
    fn from(val: &'a [u8]) -> Self {
        Self(val)
    }
}

impl<'a> From<&'a str> for Bytes<'a> {
    #[inline]
    fn from(val: &'a str) -> Self {
        Self(val.as_bytes())
    }
}

trait SearchResult: crate::pattern::SearchResult<usize> {
    /// Adjusts reject’s start position backwards to make sure it doesn’t fall
    /// withing well-formed WTF-8 sequence.
    ///
    /// Doesn’t move the start position past `begin`.  If position was adjusted,
    /// updates `*out` as well.
    fn adjust_reject_start_bwd(self, bytes: Bytes<'_>, begin: usize, out: &mut usize) -> Self;

    /// Adjusts reject’s end position forwards to make sure it doesn’t fall
    /// withing well-formed WTF-8 sequence.
    ///
    /// Doesn’t move the end position past `len`.  If position was adjusted,
    /// updates `*out` as well.
    fn adjust_reject_end_fwd(self, bytes: Bytes<'_>, len: usize, out: &mut usize) -> Self;
}

impl SearchResult for SearchStep {
    fn adjust_reject_start_bwd(mut self, bytes: Bytes<'_>, begin: usize, out: &mut usize) -> Self {
        if let SearchStep::Reject(ref mut start, _) = self {
            *start = bytes.adjust_position_bwd(begin..*start);
            *out = *start;
        }
        self
    }
    fn adjust_reject_end_fwd(mut self, bytes: Bytes<'_>, len: usize, out: &mut usize) -> Self {
        if let SearchStep::Reject(_, ref mut end) = self {
            *end = bytes.adjust_position_fwd(*end..len);
            *out = *end;
        }
        self
    }
}

impl SearchResult for MatchOnly {
    fn adjust_reject_start_bwd(self, _bytes: Bytes<'_>, _begin: usize, _out: &mut usize) -> Self {
        self
    }
    fn adjust_reject_end_fwd(self, _bytes: Bytes<'_>, _end: usize, _out: &mut usize) -> Self {
        self
    }
}

impl SearchResult for RejectOnly {
    fn adjust_reject_start_bwd(mut self, bytes: Bytes<'_>, begin: usize, out: &mut usize) -> Self {
        if let RejectOnly(Some((ref mut start, _))) = self {
            *start = bytes.adjust_position_bwd(begin..*start);
            *out = *start;
        }
        self
    }
    fn adjust_reject_end_fwd(mut self, bytes: Bytes<'_>, len: usize, out: &mut usize) -> Self {
        if let RejectOnly(Some((_, ref mut end))) = self {
            *end = bytes.adjust_position_fwd(*end..len);
            *out = *end;
        }
        self
    }
}

////////////////////////////////////////////////////////////////////////////////
// Impl for Haystack
////////////////////////////////////////////////////////////////////////////////

impl Haystack for Bytes<'_> {
    type Cursor = usize;

    fn cursor_at_front(self) -> Self::Cursor {
        0
    }
    fn cursor_at_back(self) -> Self::Cursor {
        self.0.len()
    }
    fn is_empty(self) -> bool {
        self.0.is_empty()
    }
    unsafe fn get_unchecked(self, range: Range) -> Self {
        Self(if cfg!(debug_assertions) {
            self.0.get(range).unwrap()
        } else {
            // SAFETY: Caller promises cursor is a valid split position.
            unsafe { self.0.get_unchecked(range) }
        })
    }
}

////////////////////////////////////////////////////////////////////////////////
// Impl Pattern for char
////////////////////////////////////////////////////////////////////////////////

impl<'hs> pattern::Pattern<Bytes<'hs>> for char {
    type Searcher = CharSearcher<'hs>;

    fn into_searcher(self, haystack: Bytes<'hs>) -> Self::Searcher {
        Self::Searcher::new(haystack, self)
    }

    fn is_contained_in(self, haystack: Bytes<'hs>) -> bool {
        let mut buf = [0; 4];
        encode_utf8(self, &mut buf).is_contained_in(haystack)
    }

    fn is_prefix_of(self, haystack: Bytes<'hs>) -> bool {
        let mut buf = [0; 4];
        encode_utf8(self, &mut buf).is_prefix_of(haystack)
    }
    fn strip_prefix_of(self, haystack: Bytes<'hs>) -> Option<Bytes<'hs>> {
        let mut buf = [0; 4];
        encode_utf8(self, &mut buf).strip_prefix_of(haystack)
    }

    fn is_suffix_of(self, haystack: Bytes<'hs>) -> bool {
        let mut buf = [0; 4];
        encode_utf8(self, &mut buf).is_suffix_of(haystack)
    }
    fn strip_suffix_of(self, haystack: Bytes<'hs>) -> Option<Bytes<'hs>> {
        let mut buf = [0; 4];
        encode_utf8(self, &mut buf).strip_suffix_of(haystack)
    }
}

/// Like `chr.encode_utf8(&mut buf)` but casts result to `&str`.
///
/// This is useful because we have Pattern impl for &str but not for &mut str.
fn encode_utf8(chr: char, buf: &mut [u8; 4]) -> &str {
    chr.encode_utf8(buf)
}

#[derive(Clone, Debug)]
pub struct CharSearcher<'hs> {
    haystack: Bytes<'hs>,
    state: CharSearcherState,
}

#[derive(Clone, Debug)]
struct CharSearcherState {
    /// Not yet processed range of the haystack.
    range: crate::ops::Range<usize>,
    /// Needle the searcher is looking for within the haystack.
    needle: CharBuffer,
    /// If `true` and `range` is non-empty, `haystack[range]` starts with the
    /// needle.
    is_match_fwd: bool,
    /// If `true` and `range` is non-empty, `haystack[range]` ends with the
    /// needle.
    is_match_bwd: bool,
}

impl<'hs> CharSearcher<'hs> {
    #[inline]
    pub fn new(haystack: Bytes<'hs>, chr: char) -> Self {
        Self { haystack, state: CharSearcherState::new(haystack.len(), chr) }
    }
}

unsafe impl<'hs> pattern::Searcher<Bytes<'hs>> for CharSearcher<'hs> {
    fn haystack(&self) -> Bytes<'hs> {
        self.haystack
    }

    fn next(&mut self) -> SearchStep {
        self.state.next_fwd(self.haystack)
    }
    fn next_match(&mut self) -> OptRange {
        self.state.next_fwd::<MatchOnly>(self.haystack).0
    }
    fn next_reject(&mut self) -> OptRange {
        self.state.next_fwd::<RejectOnly>(self.haystack).0
    }
}

unsafe impl<'hs> pattern::ReverseSearcher<Bytes<'hs>> for CharSearcher<'hs> {
    fn next_back(&mut self) -> SearchStep {
        self.state.next_bwd(self.haystack)
    }
    fn next_match_back(&mut self) -> OptRange {
        self.state.next_bwd::<MatchOnly>(self.haystack).0
    }
    fn next_reject_back(&mut self) -> OptRange {
        self.state.next_bwd::<RejectOnly>(self.haystack).0
    }
}

impl<'hs> pattern::DoubleEndedSearcher<Bytes<'hs>> for CharSearcher<'hs> {}

impl CharSearcherState {
    fn new(haystack_len: usize, chr: char) -> Self {
        Self {
            range: 0..haystack_len,
            needle: CharBuffer::new(chr),
            is_match_fwd: false,
            is_match_bwd: false,
        }
    }

    fn find_match_fwd(&mut self, haystack: Bytes<'_>) -> OptRange {
        let start = if take(&mut self.is_match_fwd) {
            (!self.range.is_empty()).then_some(self.range.start)
        } else {
            // SAFETY: self.range is valid range of haystack.
            let bytes = unsafe { haystack.get_unchecked(self.range.clone()) };
            // SAFETY: self.needle encodes a single character.
            unsafe { naive::find_match_fwd(bytes.as_bytes(), self.needle.as_str()) }
                .map(|pos| pos + self.range.start)
        }?;
        Some((start, start + self.needle.len()))
    }

    fn next_reject_fwd(&mut self, haystack: Bytes<'_>) -> OptRange {
        if take(&mut self.is_match_fwd) {
            if self.range.is_empty() {
                return None;
            }
            self.range.start += self.needle.len()
        }
        // SAFETY: self.range is valid range of haystack.
        let bytes = unsafe { haystack.get_unchecked(self.range.clone()) };
        if let Some(pos) = naive::find_reject_fwd(bytes.as_bytes(), self.needle.as_str()) {
            let pos = pos + self.range.start;
            let end = haystack.advance_range_start(pos..self.range.end);
            self.range.start = end;
            Some((pos, end))
        } else {
            self.range.start = self.range.end;
            None
        }
    }

    fn next_fwd<R: SearchResult>(&mut self, haystack: Bytes<'_>) -> R {
        if R::USE_EARLY_REJECT {
            match self.next_reject_fwd(haystack) {
                Some((start, end)) => R::rejecting(start, end).unwrap(),
                None => R::DONE,
            }
        } else if let Some((start, end)) = self.find_match_fwd(haystack) {
            if self.range.start < start {
                if let Some(res) = R::rejecting(self.range.start, start) {
                    self.range.start = start;
                    self.is_match_fwd = true;
                    return res;
                }
            }
            self.range.start = end;
            R::matching(start, end).unwrap()
        } else if self.range.is_empty() {
            R::DONE
        } else {
            let start = self.range.start;
            self.range.start = self.range.end;
            R::rejecting(start, self.range.end).unwrap_or(R::DONE)
        }
    }

    fn find_match_bwd(&mut self, haystack: Bytes<'_>) -> OptRange {
        let start = if take(&mut self.is_match_bwd) {
            (!self.range.is_empty()).then(|| self.range.end - self.needle.len())
        } else {
            // SAFETY: self.range is valid range of haystack.
            let bytes = unsafe { haystack.get_unchecked(self.range.clone()) };
            // SAFETY: self.needle encodes a single character.
            unsafe { naive::find_match_bwd(bytes.as_bytes(), self.needle.as_str()) }
                .map(|pos| pos + self.range.start)
        }?;
        Some((start, start + self.needle.len()))
    }

    fn next_reject_bwd(&mut self, haystack: Bytes<'_>) -> OptRange {
        if take(&mut self.is_match_bwd) {
            if self.range.is_empty() {
                return None;
            }
            self.range.end -= self.needle.len();
        }
        // SAFETY: self.range is valid range of haystack.
        let bytes = unsafe { haystack.get_unchecked(self.range.clone()) };
        if let Some(end) = naive::find_reject_bwd(bytes.as_bytes(), self.needle.as_str()) {
            let end = end + self.range.start;
            let start = haystack.advance_range_end(self.range.start..end);
            self.range.end = start;
            Some((start, end))
        } else {
            self.range.end = self.range.start;
            None
        }
    }

    fn next_bwd<R: SearchResult>(&mut self, haystack: Bytes<'_>) -> R {
        if R::USE_EARLY_REJECT {
            match self.next_reject_bwd(haystack) {
                Some((start, end)) => R::rejecting(start, end).unwrap(),
                None => R::DONE,
            }
        } else if let Some((start, end)) = self.find_match_bwd(haystack) {
            if end < self.range.end {
                if let Some(res) = R::rejecting(end, self.range.end) {
                    self.range.end = end;
                    self.is_match_bwd = true;
                    return res;
                }
            }
            self.range.end = start;
            R::matching(start, end).unwrap()
        } else if self.range.is_empty() {
            R::DONE
        } else {
            let end = self.range.end;
            self.range.end = self.range.start;
            R::rejecting(self.range.start, end).unwrap_or(R::DONE)
        }
    }
}

#[derive(Clone, Debug)]
struct CharBuffer([u8; 4], crate::num::NonZeroU8);

impl CharBuffer {
    fn new(chr: char) -> Self {
        let mut buf = [0; 4];
        let len = chr.encode_utf8(&mut buf).len();
        // SAFETY: `len` is length of a single character UTF-8 sequence.
        let len = unsafe { crate::num::NonZeroU8::new_unchecked(len as u8) };
        Self(buf, len)
    }

    fn len(&self) -> usize {
        usize::from(self.1.get())
    }

    fn as_str(&self) -> &str {
        // SAFETY: `self.0` is UTF-8 encoding of a single character and `self.1`
        // is its length.  See `new` constructor.
        unsafe { crate::str::from_utf8_unchecked(self.0.get_unchecked(..self.len())) }
    }
}

mod naive {
    use crate::slice::memchr;

    /// Looks forwards for the next position of needle within haystack.
    ///
    /// Safety: `needle` must consist of a single character.
    pub(super) unsafe fn find_match_fwd(haystack: &[u8], needle: &str) -> Option<usize> {
        debug_assert!(!needle.is_empty());
        // SAFETY: Caller promises needle is non-empty.
        let (&last_byte, head) = unsafe { needle.as_bytes().split_last().unwrap_unchecked() };
        let mut start = 0;
        while haystack.len() - start > head.len() {
            // SAFETY:
            // 1. `start` is initialised to `self.start` and only ever increased
            //    thus `self.start ≤ start`.
            // 2. We've checked `start + head.len() < haystack.len()`.
            let bytes = unsafe { haystack.get_unchecked(start + head.len()..) };
            if let Some(index) = memchr::memchr(last_byte, bytes) {
                // `start + index + head.len()` is the index of the last byte
                // thus `start + index` is the index of the first byte.
                let pos = start + index;
                // SAFETY: Since we’ve started our search with head.len()
                // offset, we know we have at least head.len() bytes in buffer.
                if unsafe { haystack.get_unchecked(pos..pos + head.len()) } == head {
                    return Some(pos);
                }
                start += index + 1;
            } else {
                break;
            }
        }
        None
    }

    /// Looks backwards for the next position of needle within haystack.
    ///
    /// Safety: `needle` must consist of a single character.
    pub(super) unsafe fn find_match_bwd(haystack: &[u8], needle: &str) -> Option<usize> {
        // SAFETY: Caller promises needle is non-empty.
        let (&first_byte, tail) = unsafe { needle.as_bytes().split_first().unwrap_unchecked() };
        let mut end = haystack.len();
        while end > tail.len() {
            // SAFETY:
            // 1. `end` is initialised to `haystack.len()` and only ever
            //    decreased thus `end ≤ haystack.len()`.
            // 2. We've checked `end > tail.len()`.
            let bytes = unsafe { haystack.get_unchecked(..end - tail.len()) };
            if let Some(pos) = memchr::memrchr(first_byte, bytes) {
                // SAFETY: Since we’ve stopped our search with tail.len()
                // offset, we know we have at least tail.len() bytes in buffer
                // after position of the byte we’ve found.
                if unsafe { haystack.get_unchecked(pos + 1..pos + 1 + tail.len()) } == tail {
                    return Some(pos);
                }
                end = pos;
            } else {
                break;
            }
        }
        None
    }

    /// Looks forwards for the next position where needle stops matching.
    ///
    /// Returns start of the next reject or `None` if there is no reject.
    pub(super) fn find_reject_fwd(haystack: &[u8], needle: &str) -> Option<usize> {
        let count =
            haystack.chunks(needle.len()).take_while(|&slice| slice == needle.as_bytes()).count();
        let start = count * needle.len();
        (start < haystack.len()).then_some(start)
    }

    /// Looks backwards for the next position where needle stops matching.
    ///
    /// Returns end of the next reject or `None` if there is no reject.
    pub(super) fn find_reject_bwd(haystack: &[u8], needle: &str) -> Option<usize> {
        debug_assert!(!needle.is_empty());
        let count =
            haystack.rchunks(needle.len()).take_while(|&slice| slice == needle.as_bytes()).count();
        let end = haystack.len() - count * needle.len();
        (end > 0).then_some(end)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Impl Pattern for FnMut(char) and FnMut(Result<char, Bytes>)
////////////////////////////////////////////////////////////////////////////////

impl<'hs, F: FnMut(char) -> bool> pattern::Pattern<Bytes<'hs>> for F {
    type Searcher = PredicateSearcher<'hs, F>;

    fn into_searcher(self, haystack: Bytes<'hs>) -> Self::Searcher {
        Self::Searcher::new(haystack, self)
    }

    fn is_prefix_of(mut self, haystack: Bytes<'hs>) -> bool {
        haystack.get_first_code_point().map_or(false, |(chr, _)| self(chr))
    }
    fn strip_prefix_of(mut self, haystack: Bytes<'hs>) -> Option<Bytes<'hs>> {
        let (chr, len) = haystack.get_first_code_point()?;
        // SAFETY: We’ve just checked slice starts with len-byte long
        // well-formed sequence.
        self(chr).then(|| unsafe { haystack.get_unchecked(len..haystack.len()) })
    }

    fn is_suffix_of(mut self, haystack: Bytes<'hs>) -> bool {
        haystack.get_last_code_point().map_or(false, |(chr, _)| self(chr))
    }
    fn strip_suffix_of(mut self, haystack: Bytes<'hs>) -> Option<Bytes<'hs>> {
        let (chr, len) = haystack.get_last_code_point()?;
        let len = haystack.len() - len;
        // SAFETY: We’ve just checked slice ends with len-byte long well-formed
        // sequence.
        self(chr).then(|| unsafe { haystack.get_unchecked(0..len) })
    }
}

#[derive(Clone, Debug)]
pub struct PredicateSearcher<'hs, F> {
    haystack: Bytes<'hs>,
    pred: F,
    start: usize,
    end: usize,
    fwd_match_len: u8,
    bwd_match_len: u8,
}

impl<'hs, F> PredicateSearcher<'hs, F> {
    fn new(haystack: Bytes<'hs>, pred: F) -> Self {
        Self { haystack, pred, start: 0, end: haystack.len(), fwd_match_len: 0, bwd_match_len: 0 }
    }
}

impl<'hs, F: FnMut(char) -> bool> PredicateSearcher<'hs, F> {
    fn find_match_fwd(&mut self) -> Option<(usize, usize)> {
        let mut start = self.start;
        while start < self.end {
            let (idx, chr, len) = self.haystack.find_code_point_fwd(start..self.end)?;
            if (self.pred)(chr) {
                return Some((idx, len));
            }
            start = idx + len;
        }
        None
    }

    fn find_match_bwd(&mut self) -> Option<(usize, usize)> {
        let mut end = self.end;
        while self.start < end {
            let (idx, chr, len) = self.haystack.find_code_point_bwd(self.start..end)?;
            if (self.pred)(chr) {
                return Some((idx, len));
            }
            end = idx;
        }
        None
    }

    fn next_fwd<R: SearchResult>(&mut self) -> R {
        while self.start < self.end {
            if self.fwd_match_len == 0 {
                let (pos, len) = self.find_match_fwd().unwrap_or((self.end, 0));
                self.fwd_match_len = len as u8;
                if pos != self.start {
                    let start = self.start;
                    self.start = pos;
                    if let Some(ret) = R::rejecting(start, pos) {
                        return ret;
                    } else if pos >= self.end {
                        break;
                    }
                }
            }

            let pos = self.start;
            self.start += usize::from(take(&mut self.fwd_match_len));
            if let Some(ret) = R::matching(pos, self.start) {
                return ret;
            }
        }
        R::DONE
    }

    fn next_bwd<R: SearchResult>(&mut self) -> R {
        while self.start < self.end {
            if self.bwd_match_len == 0 {
                let (pos, len) = self.find_match_bwd().unwrap_or((self.start, 0));
                self.bwd_match_len = len as u8;
                let pos = pos + len;
                let end = self.end;
                if pos != self.end {
                    self.end = pos;
                    if let Some(ret) = R::rejecting(pos, end) {
                        return ret;
                    } else if self.start >= self.end {
                        break;
                    }
                }
            }

            let end = self.end;
            self.end -= usize::from(take(&mut self.bwd_match_len));
            if let Some(ret) = R::matching(self.end, end) {
                return ret;
            }
        }
        R::DONE
    }
}

unsafe impl<'hs, F: FnMut(char) -> bool> Searcher<Bytes<'hs>> for PredicateSearcher<'hs, F> {
    fn haystack(&self) -> Bytes<'hs> {
        self.haystack
    }
    fn next(&mut self) -> SearchStep {
        self.next_fwd()
    }
    fn next_match(&mut self) -> OptRange {
        self.next_fwd::<MatchOnly>().0
    }
    fn next_reject(&mut self) -> OptRange {
        self.next_fwd::<RejectOnly>().0
    }
}

unsafe impl<'hs, F: FnMut(char) -> bool> pattern::ReverseSearcher<Bytes<'hs>>
    for PredicateSearcher<'hs, F>
{
    fn next_back(&mut self) -> SearchStep {
        self.next_bwd()
    }
    fn next_match_back(&mut self) -> OptRange {
        self.next_bwd::<MatchOnly>().0
    }
    fn next_reject_back(&mut self) -> OptRange {
        self.next_bwd::<RejectOnly>().0
    }
}

impl<'hs, F: FnMut(char) -> bool> pattern::DoubleEndedSearcher<Bytes<'hs>>
    for PredicateSearcher<'hs, F>
{
}

////////////////////////////////////////////////////////////////////////////////
// Impl Pattern for &str
////////////////////////////////////////////////////////////////////////////////

impl<'hs, 'p> pattern::Pattern<Bytes<'hs>> for &'p str {
    type Searcher = StrSearcher<'hs, 'p>;

    fn into_searcher(self, haystack: Bytes<'hs>) -> Self::Searcher {
        Self::Searcher::new(haystack, self)
    }

    fn is_prefix_of(self, haystack: Bytes<'hs>) -> bool {
        haystack.as_bytes().starts_with(self.as_bytes())
    }
    fn strip_prefix_of(self, haystack: Bytes<'hs>) -> Option<Bytes<'hs>> {
        haystack.as_bytes().strip_prefix(self.as_bytes()).map(Bytes)
    }

    fn is_suffix_of(self, haystack: Bytes<'hs>) -> bool {
        haystack.as_bytes().ends_with(self.as_bytes())
    }
    fn strip_suffix_of(self, haystack: Bytes<'hs>) -> Option<Bytes<'hs>> {
        haystack.as_bytes().strip_suffix(self.as_bytes()).map(Bytes)
    }
}

#[derive(Clone, Debug)]
pub struct StrSearcher<'hs, 'p> {
    haystack: Bytes<'hs>,
    state: StrSearcherInner<'p>,
}

impl<'hs, 'p> StrSearcher<'hs, 'p> {
    pub fn new(haystack: Bytes<'hs>, needle: &'p str) -> Self {
        let state = StrSearcherInner::new(haystack, needle);
        Self { haystack, state }
    }
}

unsafe impl<'hs, 'p> Searcher<Bytes<'hs>> for StrSearcher<'hs, 'p> {
    fn haystack(&self) -> Bytes<'hs> {
        self.haystack
    }
    fn next(&mut self) -> SearchStep {
        self.state.next_fwd(self.haystack)
    }
    fn next_match(&mut self) -> OptRange {
        self.state.next_fwd::<MatchOnly>(self.haystack).0
    }
    fn next_reject(&mut self) -> OptRange {
        self.state.next_fwd::<RejectOnly>(self.haystack).0
    }
}

unsafe impl<'hs, 'p> pattern::ReverseSearcher<Bytes<'hs>> for StrSearcher<'hs, 'p> {
    fn next_back(&mut self) -> SearchStep {
        self.state.next_bwd(self.haystack)
    }
    fn next_match_back(&mut self) -> OptRange {
        self.state.next_bwd::<MatchOnly>(self.haystack).0
    }
    fn next_reject_back(&mut self) -> OptRange {
        self.state.next_bwd::<RejectOnly>(self.haystack).0
    }
}

#[derive(Clone, Debug)]
enum StrSearcherInner<'p> {
    Empty(EmptySearcherState),
    Char(CharSearcherState),
    Str(StrSearcherState<'p>),
}

impl<'p> StrSearcherInner<'p> {
    fn new(haystack: Bytes<'_>, needle: &'p str) -> Self {
        let mut chars = needle.chars();
        let chr = match chars.next() {
            Some(chr) => chr,
            None => return Self::Empty(EmptySearcherState::new(haystack)),
        };
        if chars.next().is_none() {
            Self::Char(CharSearcherState::new(haystack.len(), chr))
        } else {
            Self::Str(StrSearcherState::new(haystack, needle))
        }
    }

    fn next_fwd<R: SearchResult>(&mut self, haystack: Bytes<'_>) -> R {
        match self {
            Self::Empty(state) => state.next_fwd::<R>(haystack),
            Self::Char(state) => state.next_fwd::<R>(haystack),
            Self::Str(state) => state.next_fwd::<R>(haystack),
        }
    }

    fn next_bwd<R: SearchResult>(&mut self, haystack: Bytes<'_>) -> R {
        match self {
            Self::Empty(state) => state.next_bwd::<R>(haystack),
            Self::Char(state) => state.next_bwd::<R>(haystack),
            Self::Str(state) => state.next_bwd::<R>(haystack),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Empty needle searching
////////////////////////////////////////////////////////////////////////////////

/// Empty needle rejects every character and matches every character boundary.
///
/// A character is either a well-formed WTF-8 bytes sequence or a single byte
/// whichever is longer.
#[derive(Clone, Debug)]
struct EmptySearcherState(pattern::EmptyNeedleSearcher<usize>);

impl EmptySearcherState {
    fn new(haystack: Bytes<'_>) -> Self {
        Self(pattern::EmptyNeedleSearcher::new(haystack))
    }

    fn next_fwd<R: pattern::SearchResult>(&mut self, bytes: Bytes<'_>) -> R {
        self.0.next_fwd(|range| bytes.advance_range_start(range))
    }

    fn next_bwd<R: pattern::SearchResult>(&mut self, bytes: Bytes<'_>) -> R {
        self.0.next_bwd(|range| bytes.advance_range_end(range))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Full substring search
////////////////////////////////////////////////////////////////////////////////

/// A substring search.
#[derive(Clone, Debug)]
struct StrSearcherState<'p> {
    needle: &'p str,
    searcher: TwoWaySearcher,
}

impl<'p> StrSearcherState<'p> {
    fn new(haystack: Bytes<'_>, needle: &'p str) -> Self {
        let searcher = TwoWaySearcher::new(haystack.len(), needle.as_bytes());
        Self { needle, searcher }
    }

    fn next_fwd<R: SearchResult>(&mut self, bytes: Bytes<'_>) -> R {
        if self.searcher.position >= bytes.len() {
            return R::DONE;
        }
        if self.searcher.memory == usize::MAX {
            self.searcher.next_fwd::<R>(bytes.0, self.needle.as_bytes(), true)
        } else {
            self.searcher.next_fwd::<R>(bytes.0, self.needle.as_bytes(), false)
        }
        .adjust_reject_end_fwd(bytes, bytes.len(), &mut self.searcher.position)
    }

    fn next_bwd<R: SearchResult>(&mut self, bytes: Bytes<'_>) -> R {
        if self.searcher.end == 0 {
            return R::DONE;
        }
        if self.searcher.memory == usize::MAX {
            self.searcher.next_bwd::<R>(bytes.0, self.needle.as_bytes(), true)
        } else {
            self.searcher.next_bwd::<R>(bytes.0, self.needle.as_bytes(), false)
        }
        .adjust_reject_start_bwd(bytes, 0, &mut self.searcher.end)
    }
}

/// The internal state of the two-way substring search algorithm.
#[derive(Clone, Debug)]
struct TwoWaySearcher {
    // constants
    /// critical factorization index
    crit_pos: usize,
    /// critical factorization index for reversed needle
    crit_pos_back: usize,
    period: usize,
    /// `byteset` is an extension (not part of the two way algorithm);
    /// it's a 64-bit "fingerprint" where each set bit `j` corresponds
    /// to a (byte & 63) == j present in the needle.
    byteset: u64,

    // variables
    position: usize,
    end: usize,
    /// index into needle before which we have already matched
    memory: usize,
    /// index into needle after which we have already matched
    memory_back: usize,
}

/*
    This is the Two-Way search algorithm, which was introduced in the paper:
    Crochemore, M., Perrin, D., 1991, Two-way string-matching, Journal of the ACM 38(3):651-675.

    Here's some background information.

    A *word* is a string of symbols. The *length* of a word should be a familiar
    notion, and here we denote it for any word x by |x|.
    (We also allow for the possibility of the *empty word*, a word of length zero).

    If x is any non-empty word, then an integer p with 0 < p <= |x| is said to be a
    *period* for x iff for all i with 0 <= i <= |x| - p - 1, we have x[i] == x[i+p].
    For example, both 1 and 2 are periods for the string "aa". As another example,
    the only period of the string "abcd" is 4.

    We denote by period(x) the *smallest* period of x (provided that x is non-empty).
    This is always well-defined since every non-empty word x has at least one period,
    |x|. We sometimes call this *the period* of x.

    If u, v and x are words such that x = uv, where uv is the concatenation of u and
    v, then we say that (u, v) is a *factorization* of x.

    Let (u, v) be a factorization for a word x. Then if w is a non-empty word such
    that both of the following hold

      - either w is a suffix of u or u is a suffix of w
      - either w is a prefix of v or v is a prefix of w

    then w is said to be a *repetition* for the factorization (u, v).

    Just to unpack this, there are four possibilities here. Let w = "abc". Then we
    might have:

      - w is a suffix of u and w is a prefix of v. ex: ("lolabc", "abcde")
      - w is a suffix of u and v is a prefix of w. ex: ("lolabc", "ab")
      - u is a suffix of w and w is a prefix of v. ex: ("bc", "abchi")
      - u is a suffix of w and v is a prefix of w. ex: ("bc", "a")

    Note that the word vu is a repetition for any factorization (u,v) of x = uv,
    so every factorization has at least one repetition.

    If x is a string and (u, v) is a factorization for x, then a *local period* for
    (u, v) is an integer r such that there is some word w such that |w| = r and w is
    a repetition for (u, v).

    We denote by local_period(u, v) the smallest local period of (u, v). We sometimes
    call this *the local period* of (u, v). Provided that x = uv is non-empty, this
    is well-defined (because each non-empty word has at least one factorization, as
    noted above).

    It can be proven that the following is an equivalent definition of a local period
    for a factorization (u, v): any positive integer r such that x[i] == x[i+r] for
    all i such that |u| - r <= i <= |u| - 1 and such that both x[i] and x[i+r] are
    defined. (i.e., i > 0 and i + r < |x|).

    Using the above reformulation, it is easy to prove that

        1 <= local_period(u, v) <= period(uv)

    A factorization (u, v) of x such that local_period(u,v) = period(x) is called a
    *critical factorization*.

    The algorithm hinges on the following theorem, which is stated without proof:

    **Critical Factorization Theorem** Any word x has at least one critical
    factorization (u, v) such that |u| < period(x).

    The purpose of maximal_suffix is to find such a critical factorization.

    If the period is short, compute another factorization x = u' v' to use
    for reverse search, chosen instead so that |v'| < period(x).

*/
impl TwoWaySearcher {
    fn new(haystack_len: usize, needle: &[u8]) -> TwoWaySearcher {
        let (crit_pos_false, period_false) = TwoWaySearcher::maximal_suffix(needle, false);
        let (crit_pos_true, period_true) = TwoWaySearcher::maximal_suffix(needle, true);

        let (crit_pos, period) = if crit_pos_false > crit_pos_true {
            (crit_pos_false, period_false)
        } else {
            (crit_pos_true, period_true)
        };

        // A particularly readable explanation of what's going on here can be found
        // in Crochemore and Rytter's book "Text Algorithms", ch 13. Specifically
        // see the code for "Algorithm CP" on p. 323.
        //
        // What's going on is we have some critical factorization (u, v) of the
        // needle, and we want to determine whether u is a suffix of
        // &v[..period]. If it is, we use "Algorithm CP1". Otherwise we use
        // "Algorithm CP2", which is optimized for when the period of the needle
        // is large.
        if needle[..crit_pos] == needle[period..period + crit_pos] {
            // short period case -- the period is exact
            // compute a separate critical factorization for the reversed needle
            // x = u' v' where |v'| < period(x).
            //
            // This is sped up by the period being known already.
            // Note that a case like x = "acba" may be factored exactly forwards
            // (crit_pos = 1, period = 3) while being factored with approximate
            // period in reverse (crit_pos = 2, period = 2). We use the given
            // reverse factorization but keep the exact period.
            let crit_pos_back = needle.len()
                - cmp::max(
                    TwoWaySearcher::reverse_maximal_suffix(needle, period, false),
                    TwoWaySearcher::reverse_maximal_suffix(needle, period, true),
                );

            TwoWaySearcher {
                crit_pos,
                crit_pos_back,
                period,
                byteset: Self::byteset_create(&needle[..period]),

                position: 0,
                end: haystack_len,
                memory: 0,
                memory_back: needle.len(),
            }
        } else {
            // long period case -- we have an approximation to the actual period,
            // and don't use memorization.
            //
            // Approximate the period by lower bound max(|u|, |v|) + 1.
            // The critical factorization is efficient to use for both forward and
            // reverse search.

            TwoWaySearcher {
                crit_pos,
                crit_pos_back: crit_pos,
                period: cmp::max(crit_pos, needle.len() - crit_pos) + 1,
                byteset: Self::byteset_create(needle),

                position: 0,
                end: haystack_len,
                memory: usize::MAX, // Dummy value to signify that the period is long
                memory_back: usize::MAX,
            }
        }
    }

    #[inline]
    fn byteset_create(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0, |a, &b| (1 << (b & 0x3f)) | a)
    }

    #[inline]
    fn byteset_contains(&self, byte: u8) -> bool {
        (self.byteset >> ((byte & 0x3f) as usize)) & 1 != 0
    }

    // One of the main ideas of Two-Way is that we factorize the needle into
    // two halves, (u, v), and begin trying to find v in the haystack by scanning
    // left to right. If v matches, we try to match u by scanning right to left.
    // How far we can jump when we encounter a mismatch is all based on the fact
    // that (u, v) is a critical factorization for the needle.
    #[inline]
    fn next_fwd<R: SearchResult>(
        &mut self,
        haystack: &[u8],
        needle: &[u8],
        long_period: bool,
    ) -> R {
        // `next()` uses `self.position` as its cursor
        let old_pos = self.position;
        let needle_last = needle.len() - 1;
        'search: loop {
            // Check that we have room to search in
            // position + needle_last can not overflow if we assume slices
            // are bounded by isize's range.
            let tail_byte = match haystack.get(self.position + needle_last) {
                Some(&b) => b,
                None => {
                    self.position = haystack.len();
                    return R::rejecting(old_pos, self.position).unwrap_or(R::DONE);
                }
            };

            if old_pos != self.position {
                if let Some(ret) = R::rejecting(old_pos, self.position) {
                    return ret;
                }
            }

            // Quickly skip by large portions unrelated to our substring
            if !self.byteset_contains(tail_byte) {
                self.position += needle.len();
                if !long_period {
                    self.memory = 0;
                }
                continue 'search;
            }

            // See if the right part of the needle matches
            let start =
                if long_period { self.crit_pos } else { cmp::max(self.crit_pos, self.memory) };
            for i in start..needle.len() {
                if needle[i] != haystack[self.position + i] {
                    self.position += i - self.crit_pos + 1;
                    if !long_period {
                        self.memory = 0;
                    }
                    continue 'search;
                }
            }

            // See if the left part of the needle matches
            let start = if long_period { 0 } else { self.memory };
            for i in (start..self.crit_pos).rev() {
                if needle[i] != haystack[self.position + i] {
                    self.position += self.period;
                    if !long_period {
                        self.memory = needle.len() - self.period;
                    }
                    continue 'search;
                }
            }

            // We have found a match!
            let match_pos = self.position;

            // Note: add self.period instead of needle.len() to have overlapping matches
            self.position += needle.len();
            if !long_period {
                self.memory = 0; // set to needle.len() - self.period for overlapping matches
            }

            if let Some(ret) = R::matching(match_pos, match_pos + needle.len()) {
                return ret;
            }
        }
    }

    // Follows the ideas in `next()`.
    //
    // The definitions are symmetrical, with period(x) = period(reverse(x))
    // and local_period(u, v) = local_period(reverse(v), reverse(u)), so if (u, v)
    // is a critical factorization, so is (reverse(v), reverse(u)).
    //
    // For the reverse case we have computed a critical factorization x = u' v'
    // (field `crit_pos_back`). We need |u| < period(x) for the forward case and
    // thus |v'| < period(x) for the reverse.
    //
    // To search in reverse through the haystack, we search forward through
    // a reversed haystack with a reversed needle, matching first u' and then v'.
    #[inline]
    fn next_bwd<R: SearchResult>(
        &mut self,
        haystack: &[u8],
        needle: &[u8],
        long_period: bool,
    ) -> R {
        // `next_back()` uses `self.end` as its cursor -- so that `next()` and `next_back()`
        // are independent.
        let old_end = self.end;
        'search: loop {
            // Check that we have room to search in
            // end - needle.len() will wrap around when there is no more room,
            // but due to slice length limits it can never wrap all the way back
            // into the length of haystack.
            let front_byte = match haystack.get(self.end.wrapping_sub(needle.len())) {
                Some(&b) => b,
                None => {
                    self.end = 0;
                    return R::rejecting(0, old_end).unwrap_or(R::DONE);
                }
            };

            if old_end != self.end {
                if let Some(ret) = R::rejecting(self.end, old_end) {
                    return ret;
                }
            }

            // Quickly skip by large portions unrelated to our substring
            if !self.byteset_contains(front_byte) {
                self.end -= needle.len();
                if !long_period {
                    self.memory_back = needle.len();
                }
                continue 'search;
            }

            // See if the left part of the needle matches
            let crit = if long_period {
                self.crit_pos_back
            } else {
                cmp::min(self.crit_pos_back, self.memory_back)
            };
            for i in (0..crit).rev() {
                if needle[i] != haystack[self.end - needle.len() + i] {
                    self.end -= self.crit_pos_back - i;
                    if !long_period {
                        self.memory_back = needle.len();
                    }
                    continue 'search;
                }
            }

            // See if the right part of the needle matches
            let needle_end = if long_period { needle.len() } else { self.memory_back };
            for i in self.crit_pos_back..needle_end {
                if needle[i] != haystack[self.end - needle.len() + i] {
                    self.end -= self.period;
                    if !long_period {
                        self.memory_back = self.period;
                    }
                    continue 'search;
                }
            }

            // We have found a match!
            let match_pos = self.end - needle.len();
            // Note: sub self.period instead of needle.len() to have overlapping matches
            self.end -= needle.len();
            if !long_period {
                self.memory_back = needle.len();
            }

            if let Some(ret) = R::matching(match_pos, match_pos + needle.len()) {
                return ret;
            }
        }
    }

    // Compute the maximal suffix of `arr`.
    //
    // The maximal suffix is a possible critical factorization (u, v) of `arr`.
    //
    // Returns (`i`, `p`) where `i` is the starting index of v and `p` is the
    // period of v.
    //
    // `order_greater` determines if lexical order is `<` or `>`. Both
    // orders must be computed -- the ordering with the largest `i` gives
    // a critical factorization.
    //
    // For long period cases, the resulting period is not exact (it is too short).
    #[inline]
    fn maximal_suffix(arr: &[u8], order_greater: bool) -> (usize, usize) {
        let mut left = 0; // Corresponds to i in the paper
        let mut right = 1; // Corresponds to j in the paper
        let mut offset = 0; // Corresponds to k in the paper, but starting at 0
        // to match 0-based indexing.
        let mut period = 1; // Corresponds to p in the paper

        while let Some(&a) = arr.get(right + offset) {
            // `left` will be inbounds when `right` is.
            let b = arr[left + offset];
            if (a < b && !order_greater) || (a > b && order_greater) {
                // Suffix is smaller, period is entire prefix so far.
                right += offset + 1;
                offset = 0;
                period = right - left;
            } else if a == b {
                // Advance through repetition of the current period.
                if offset + 1 == period {
                    right += offset + 1;
                    offset = 0;
                } else {
                    offset += 1;
                }
            } else {
                // Suffix is larger, start over from current location.
                left = right;
                right += 1;
                offset = 0;
                period = 1;
            }
        }
        (left, period)
    }

    // Compute the maximal suffix of the reverse of `arr`.
    //
    // The maximal suffix is a possible critical factorization (u', v') of `arr`.
    //
    // Returns `i` where `i` is the starting index of v', from the back;
    // returns immediately when a period of `known_period` is reached.
    //
    // `order_greater` determines if lexical order is `<` or `>`. Both
    // orders must be computed -- the ordering with the largest `i` gives
    // a critical factorization.
    //
    // For long period cases, the resulting period is not exact (it is too short).
    fn reverse_maximal_suffix(arr: &[u8], known_period: usize, order_greater: bool) -> usize {
        let mut left = 0; // Corresponds to i in the paper
        let mut right = 1; // Corresponds to j in the paper
        let mut offset = 0; // Corresponds to k in the paper, but starting at 0
        // to match 0-based indexing.
        let mut period = 1; // Corresponds to p in the paper
        let n = arr.len();

        while right + offset < n {
            let a = arr[n - (1 + right + offset)];
            let b = arr[n - (1 + left + offset)];
            if (a < b && !order_greater) || (a > b && order_greater) {
                // Suffix is smaller, period is entire prefix so far.
                right += offset + 1;
                offset = 0;
                period = right - left;
            } else if a == b {
                // Advance through repetition of the current period.
                if offset + 1 == period {
                    right += offset + 1;
                    offset = 0;
                } else {
                    offset += 1;
                }
            } else {
                // Suffix is larger, start over from current location.
                left = right;
                right += 1;
                offset = 0;
                period = 1;
            }
            if period == known_period {
                break;
            }
        }
        debug_assert!(period <= known_period);
        left
    }
}
