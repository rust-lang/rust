//! This module contains types and functions to support formatting specific line ranges.
use std::{cmp, iter, str};

use itertools::Itertools;
use multimap::MultiMap;
use rustc_serialize::{self, json};

use codemap::LineRange;

/// A range that is inclusive of both ends.
#[derive(Clone, Copy, Debug, Eq, PartialEq, RustcDecodable)]
struct Range {
    pub lo: usize,
    pub hi: usize,
}

impl<'a> From<&'a LineRange> for Range {
    fn from(range: &'a LineRange) -> Range {
        Range::new(range.lo, range.hi)
    }
}

impl Range {
    fn new(lo: usize, hi: usize) -> Range {
        Range { lo: lo, hi: hi }
    }

    fn is_empty(self) -> bool {
        self.lo > self.hi
    }

    fn contains(self, other: Range) -> bool {
        if other.is_empty() {
            true
        } else {
            !self.is_empty() && self.lo <= other.lo && self.hi >= other.hi
        }
    }

    fn intersects(self, other: Range) -> bool {
        if self.is_empty() || other.is_empty() {
            false
        } else {
            (self.lo <= other.hi && other.hi <= self.hi) ||
            (other.lo <= self.hi && self.hi <= other.hi)
        }
    }

    fn adjacent_to(self, other: Range) -> bool {
        if self.is_empty() || other.is_empty() {
            false
        } else {
            self.hi + 1 == other.lo || other.hi + 1 == self.lo
        }
    }

    /// Returns a new `Range` with lines from `self` and `other` if they were adjacent or
    /// intersect; returns `None` otherwise.
    fn merge(self, other: Range) -> Option<Range> {
        if self.adjacent_to(other) || self.intersects(other) {
            Some(Range::new(cmp::min(self.lo, other.lo), cmp::max(self.hi, other.hi)))
        } else {
            None
        }
    }
}

/// A set of lines in files.
///
/// It is represented as a multimap keyed on file names, with values a collection of
/// non-overlapping ranges sorted by their start point. An inner `None` is interpreted to mean all
/// lines in all files.
#[derive(Clone, Debug, Default)]
pub struct FileLines(Option<MultiMap<String, Range>>);

/// Normalizes the ranges so that the invariants for `FileLines` hold: ranges are non-overlapping,
/// and ordered by their start point.
fn normalize_ranges(map: &mut MultiMap<String, Range>) {
    for (_, ranges) in map.iter_all_mut() {
        ranges.sort_by_key(|x| x.lo);
        let merged = ranges.drain(..).coalesce(|x, y| x.merge(y).ok_or((x, y))).collect();
        *ranges = merged;
    }
}

impl FileLines {
    /// Creates a `FileLines` that contains all lines in all files.
    pub fn all() -> FileLines {
        FileLines(None)
    }

    /// Creates a `FileLines` from a `MultiMap`, ensuring that the invariants hold.
    fn from_multimap(map: MultiMap<String, Range>) -> FileLines {
        let mut map = map;
        normalize_ranges(&mut map);
        FileLines(Some(map))
    }

    /// Returns an iterator over the files contained in `self`.
    pub fn files(&self) -> Files {
        Files(self.0.as_ref().map(MultiMap::keys))
    }

    /// Returns true if `range` is fully contained in `self`.
    pub fn contains(&self, range: &LineRange) -> bool {
        let map = match self.0 {
            // `None` means "all lines in all files".
            None => return true,
            Some(ref map) => map,
        };

        match map.get_vec(range.file_name()) {
            None => false,
            Some(ranges) => ranges.iter().any(|r| r.contains(Range::from(range))),
        }
    }

    /// Returns true if any lines in `range` are in `self`.
    pub fn intersects(&self, range: &LineRange) -> bool {
        let map = match self.0 {
            // `None` means "all lines in all files".
            None => return true,
            Some(ref map) => map,
        };

        match map.get_vec(range.file_name()) {
            None => false,
            Some(ranges) => ranges.iter().any(|r| r.intersects(Range::from(range))),
        }
    }
}

/// FileLines files iterator.
pub struct Files<'a>(Option<::std::collections::hash_map::Keys<'a, String, Vec<Range>>>);

impl<'a> iter::Iterator for Files<'a> {
    type Item = &'a String;

    fn next(&mut self) -> Option<&'a String> {
        self.0.as_mut().and_then(Iterator::next)
    }
}

// This impl is needed for `Config::override_value` to work for use in tests.
impl str::FromStr for FileLines {
    type Err = String;

    fn from_str(s: &str) -> Result<FileLines, String> {
        let v: Vec<JsonSpan> = try!(json::decode(s).map_err(|e| e.to_string()));
        let m = v.into_iter().map(JsonSpan::into_tuple).collect();
        Ok(FileLines::from_multimap(m))
    }
}

// For JSON decoding.
#[derive(Clone, Debug, RustcDecodable)]
struct JsonSpan {
    file: String,
    range: (usize, usize),
}

impl JsonSpan {
    // To allow `collect()`ing into a `MultiMap`.
    fn into_tuple(self) -> (String, Range) {
        let (lo, hi) = self.range;
        (self.file, Range::new(lo, hi))
    }
}

// This impl is needed for inclusion in the `Config` struct. We don't have a toml representation
// for `FileLines`, so it will just panic instead.
impl rustc_serialize::Decodable for FileLines {
    fn decode<D: rustc_serialize::Decoder>(_: &mut D) -> Result<Self, D::Error> {
        unimplemented!();
    }
}

#[cfg(test)]
mod test {
    use super::Range;

    #[test]
    fn test_range_intersects() {
        assert!(Range::new(1, 2).intersects(Range::new(1, 1)));
        assert!(Range::new(1, 2).intersects(Range::new(2, 2)));
        assert!(!Range::new(1, 2).intersects(Range::new(0, 0)));
        assert!(!Range::new(1, 2).intersects(Range::new(3, 10)));
        assert!(!Range::new(1, 3).intersects(Range::new(5, 5)));
    }

    #[test]
    fn test_range_adjacent_to() {
        assert!(!Range::new(1, 2).adjacent_to(Range::new(1, 1)));
        assert!(!Range::new(1, 2).adjacent_to(Range::new(2, 2)));
        assert!(Range::new(1, 2).adjacent_to(Range::new(0, 0)));
        assert!(Range::new(1, 2).adjacent_to(Range::new(3, 10)));
        assert!(!Range::new(1, 3).adjacent_to(Range::new(5, 5)));
    }

    #[test]
    fn test_range_contains() {
        assert!(Range::new(1, 2).contains(Range::new(1, 1)));
        assert!(Range::new(1, 2).contains(Range::new(2, 2)));
        assert!(!Range::new(1, 2).contains(Range::new(0, 0)));
        assert!(!Range::new(1, 2).contains(Range::new(3, 10)));
    }

    #[test]
    fn test_range_merge() {
        assert_eq!(None, Range::new(1, 3).merge(Range::new(5, 5)));
        assert_eq!(None, Range::new(4, 7).merge(Range::new(0, 1)));
        assert_eq!(Some(Range::new(3, 7)),
                   Range::new(3, 5).merge(Range::new(4, 7)));
        assert_eq!(Some(Range::new(3, 7)),
                   Range::new(3, 5).merge(Range::new(5, 7)));
        assert_eq!(Some(Range::new(3, 7)),
                   Range::new(3, 5).merge(Range::new(6, 7)));
        assert_eq!(Some(Range::new(3, 7)),
                   Range::new(3, 7).merge(Range::new(4, 5)));
    }
}
